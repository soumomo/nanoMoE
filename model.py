#imports
from torch.autograd import grad_mode
import torch
import torch.nn as nn
from einops import repeat, unpack, rearrange, einsum

class Linear(nn.Module):
    def __init__(self , d_in: int , d_out: int):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_out , d_in))
        variance = 2 / (d_out + d_in)
        std = variance ** 0.5

        nn.init.trunc_normal_(self.weight , mean=0.0, std=std, a=-2.0*std, b=2.0*std)
    
    def forward(self , x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x , self.weight.t())


class Embedding(nn.Module):
    def __init__(self , vocab_size: int , d_model :int):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        #defining the weight matrix
        self.weight = nn.Parameter(torch.empty(vocab_size , d_model))

        variance = 1
        std = variance ** 0.5
        nn.init.trunc_normal_(self.weight , mean=0.0, std=std, a=-2.0*std, b=2.0*std)


    def forward(self ,token_ids: torch.Tensor ) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self , d_model: int , eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.d_model = d_model

        self.weight = nn.Parameter(torch.ones(d_model ,))

    def forward(self , x: torch.Tensor):
        orig_dtype = x.dtype
        x = x.float()
        return ((x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))*self.weight).to(orig_dtype)

class SwiGLU(nn.Module):
    def __init__(self , d_model: int ,d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model , d_ff)
        self.w2 = Linear(d_ff , d_model)
        self.w3 = Linear(d_model , d_ff)

    def forward(self , x):
        out1 = self.w1(x)
        out3 = self.w3(x)
        silu_out1 = out1 * torch.sigmoid(out1)
        gated_mul = silu_out1 * out3
        return self.w2(gated_mul)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_val)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

class RoPE(nn.Module):
    def __init__(self , d_k: int , max_seq_len: int , theta: float):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

        # compute position vectors
        # create a 1D tensor containing values from 0 to max_seq_len - 1
        positions = torch.arange(max_seq_len)

        # compute frequency bands
        pair_idx = torch.arange(0 , d_k , 2).float()
        freq = self.theta ** (-pair_idx/d_k)

        #calculate the angles
        angles = torch.outer(positions , freq)

        #storing cached
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)

        #register as buffer
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """ 
        indexing the cox/sin cache
        precomputed lookup table of shape (batch, seq_len, d_k // 2)
        """
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        #broadcasting
        """
        if the input /key tensor x has shape [batch, heads, seq_len, d_k] then our cos/sin are missing the heads dimension
        we insert a dimension of 1 before the seq_len dimension so that it becomes (batch, 1, seq_len, d_k // 2) and can broadcasr over any head
        """
        #this hardcoded is giving dimension mismatch
        # cos = cos.unsqueeze(1)
        # sin = sin.unsqueeze(1)

        #initiate the dimension with all 1s
        view_shape = [1]*x.ndim

        #fill in the sequence length and head dimension
        view_shape[-1] = x.shape[-1] // 2
        view_shape[-2] = x.shape[-2]

        if token_positions.ndim > 1:
            view_shape[0] = token_positions.shape[0]

        cos = cos.view(view_shape)
        sin = sin.view(view_shape)

        #performing pair wise rotation
        #sliced into even and odd components
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        #compute the rotated components
        x_even_rotated = x_even*cos - x_odd*sin
        x_odd_rotated = x_odd*cos + x_even*sin

        #reassembling the final output tensor
        out = torch.empty_like(x)
        out[..., ::2] = x_even_rotated
        out[..., 1::2] = x_odd_rotated

        return out


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        #compute scaled dot product
        K_t = K.transpose(-1,-2)
        scaled_dot_product = torch.matmul(Q , K_t)/(d_k ** 0.5)

        if mask is not None:
        # anywhere mask is False (or 0), fill with negative infinity
            scaled_dot_product = scaled_dot_product.masked_fill(mask == False, float('-inf'))

        attn_weights = softmax(scaled_dot_product , dim = -1)
        output = torch.matmul(attn_weights , V)


        return output

class Multi_Head_Attention(nn.Module):
    '''
    there will be two versions —— with and without RoPE
    '''
    def __init__(self, d_model: int , num_heads: int , max_seq_len: int|None = None , theta: float | None = None):
        super().__init__()
        self.q_proj = Linear(d_model , d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.attention = Attention()
        self.num_heads = num_heads                              
        self.d_k = d_model//num_heads
        if max_seq_len is not None:
            self.rope = RoPE(d_model//num_heads, max_seq_len, theta)

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # (batch_size, seq_len, d_model))
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # reshape and transpose
        Q = Q.view(*Q.shape[:-1], self.num_heads , self.d_k).transpose(-3,-2)
        K = K.view(*K.shape[:-1], self.num_heads , self.d_k).transpose(-3,-2)
        V = V.view(*V.shape[:-1], self.num_heads , self.d_k).transpose(-3,-2)

        # new shape for Q, K, V: [..., num_heads, seq_len, d_k]

        if hasattr(self , 'rope'):
            Q = self.rope(Q , token_positions )
            K = self.rope(K , token_positions )

        # causal masking time
        # torch.tril creates a lower triangular matrix of 1s (past positions) and 0s (future positions)
        seq_len = x.shape[-2]
        mask  = torch.tril(torch.ones(seq_len , seq_len , device=x.device , dtype=bool))
        out =  self.attention(Q , K , V , mask) #(... , num_heads , seq_len , d_k)
        out = out.transpose(-3,-2)
        out = out.reshape(*out.shape[:-2] , -1)
        return self.output_proj(out) 

class TransformerBlock(nn.Module):
    '''
    Sub-Layer 1: Multi-Head Self-Attention

    x → RMSNorm → Multi_Head_Attention → + x  →  z
                                      ↑
                              (residual skip)

    Sub-Layer 2: Feed-Forward Network (SwiGLU)

    z → RMSNorm → SwiGLU → + z  →  output
                         ↑
                 (residual skip)


    
    '''

    def __init__(self ,d_model, num_heads , d_ff, max_seq_len, theta ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = Multi_Head_Attention(d_model , num_heads ,max_seq_len ,theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model , d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        norm_x = self.ln1(x)
        attn_layer = self.attn(norm_x , token_positions)
        z = x + attn_layer


        norm_z = self.ln2(z)
        passed_z = self.ffn(norm_z)
        out = z + passed_z

        return out
        

class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        num_experts: int = 4,
        top_k: int = 2
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                num_experts=num_experts,
                top_k=top_k
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        seq_len = x.shape[-1]
        token_positions = torch.arange(seq_len, device=x.device)
        
        h = self.token_embeddings(x)
        
        # accumulate aux_loss across MoE layers
        total_aux_loss = 0.0
        for layer in self.layers:
            h, aux_loss = layer(h, token_positions)
            total_aux_loss += aux_loss
            
        h = self.ln_final(h)
        logits = self.lm_head(h)
        
        # compute loss if targets provided during training
        loss = None
        if targets is not None:
            import torch.nn.functional as F
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce_loss + 0.01 * total_aux_loss
            
        return logits, loss, total_aux_loss 

    
    

class GQA(nn.Module):
    '''
    there will be two versions —— with and without RoPE
    '''
    def __init__(self, d_model: int , num_heads: int, num_kv_heads: int , max_seq_len: int|None = None , theta: float | None = None):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv_group = num_heads//num_kv_heads
        kv_dim = d_model//self.num_queries_per_kv_group
        self.q_proj = Linear(d_model , d_model)
        self.k_proj = Linear(d_model, kv_dim)
        self.v_proj = Linear(d_model, kv_dim)

        self.output_proj = Linear(d_model, d_model)
        self.attention = Attention()
        self.num_heads = num_heads
        self.d_k = d_model//num_heads
        if max_seq_len is not None:
            self.rope = RoPE(d_model//num_heads, max_seq_len, theta)

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # (batch_size, seq_len, d_model))
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # reshape and transpose
        Q = Q.view(*Q.shape[:-1], self.num_heads , self.d_k).transpose(-3,-2)
        K = K.view(*K.shape[:-1], self.num_kv_heads , self.d_k).transpose(-3,-2)
        V = V.view(*V.shape[:-1], self.num_kv_heads , self.d_k).transpose(-3,-2)

        

        if hasattr(self , 'rope'):
            Q = self.rope(Q , token_positions )
            K = self.rope(K , token_positions )

        if self.num_queries_per_kv_group > 1:
            K = repeat(K, 'b g s d -> b (g r) s d', r=self.num_queries_per_kv_group)
            V = repeat(V, 'b g s d -> b (g r) s d', r=self.num_queries_per_kv_group)


        # causal masking time
        # torch.tril creates a lower triangular matrix of 1s (past positions) and 0s (future positions)
        seq_len = x.shape[-2]
        mask  = torch.tril(torch.ones(seq_len , seq_len , device=x.device , dtype=bool))
        out =  self.attention(Q , K , V , mask) #(... , num_heads , seq_len , d_k)
        out = out.transpose(-3,-2)
        out = out.reshape(*out.shape[:-2] , -1)
        return self.output_proj(out) 


class NaiveMLA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, latent_dim_q: int,latent_dim_kv: int, d_v: int,d_kC: int, d_kR: int,  max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.latent_dim_q = latent_dim_q
        self.latent_dim_kv = latent_dim_kv
        self.d_v = d_v
        self.d_kC = d_kC
        self.d_kR = d_kR
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_k = d_model//num_heads
        self.attention = Attention()

        self.W_DQ = Linear(d_model, latent_dim_q) #compresses query
        self.W_UQ = Linear(latent_dim_q, num_heads*(d_kC + d_kR)) #up project query

        self.W_DKV = Linear(d_model, latent_dim_kv) #compress kv
        self.W_UK = Linear(latent_dim_kv, num_heads*self.d_kC) #key up-proj
        self.W_KR = Linear(d_model, d_kR) #decoupled rope key
        self.W_UV = Linear(latent_dim_kv, num_heads*d_v) #value up proj

        self.out_proj = Linear(num_heads*d_v, d_model) #final o/p    
  
        self.rope = RoPE(d_kR, max_seq_len, theta) #rope for decoupled parts only

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        L_kv = self.W_DKV(x) #(batch_size, seq_len, latent_dim_kv)
        L_q = self.W_DQ(x) #(batch_size, seq_len, latent_dim_q)
        Q = self.W_UQ(L_q) #(b, t, num_heads*(d_kC + d_kR))

        [Q_C_packed, Q_R_packed] = unpack(Q, [self.num_heads * self.d_kC, self.num_heads * self.d_kR], "b t *")
        Q_C = rearrange(Q_C_packed, "b t (h d_c) -> b h t d_c", h = self.num_heads)
        Q_R = rearrange(Q_R_packed, "b t (h d_r) -> b h t d_r", h = self.num_heads)

        # [b, t, latent_dim_kv] -> [b, t, h* d_kc] -> [b, h, t, d_kc]
        K_C = rearrange(self.W_UK(L_kv), 'b t (h d) -> b h t d', h=self.num_heads)
        # (b, t, d_kR) -> (b 1 t d_kR)
        K_R = rearrange(self.W_KR(x), "b t d -> b 1 t d")


        V = rearrange(self.W_UV(L_kv), "b t (h d_v) -> b h t d_v", h = self.num_heads)

        # pass through rope 
        Q_R = self.rope(Q_R, token_positions)
        K_R = self.rope(K_R, token_positions)

        # torch.tril creates a lower triangular matrix of 1s (past positions) and 0s (future positions)
        seq_len = x.shape[-2]
        mask  = torch.tril(torch.ones(seq_len , seq_len , device=x.device , dtype=bool))

        Q = torch.cat([Q_C, Q_R], dim = -1)
        #(b 1 t d_kR -> b h t d_kR)
        K_R_expand = repeat(K_R, "b 1 t d -> b h t d", h=self.num_heads)
        K = torch.cat([K_C, K_R_expand], dim = -1)

        out =  self.attention(Q , K , V , mask) #(b, h, t, d_v)
        out_flattened = rearrange(out, 'b h t d -> b t (h d)')
        final_output = self.out_proj(out_flattened)
        return final_output
    
class MLA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, latent_dim_q: int,latent_dim_kv: int, d_v: int,d_kC: int, d_kR: int,  max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.latent_dim_q = latent_dim_q
        self.latent_dim_kv = latent_dim_kv
        self.d_v = d_v
        self.d_kC = d_kC
        self.d_kR = d_kR
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_k = d_model//num_heads

        self.W_DQ = Linear(d_model, latent_dim_q) #compresses query
        self.W_UQ = Linear(latent_dim_q, num_heads*(d_kC + d_kR)) #up project query

        self.W_DKV = Linear(d_model, latent_dim_kv) #compress kv

        self.W_KR = Linear(d_model, d_kR) #decoupled rope key

        self.W_UK = nn.Parameter(torch.empty(num_heads, d_kC, latent_dim_kv)) #key up-proj
        self.W_UV = nn.Parameter(torch.empty(num_heads, d_v, latent_dim_kv)) #value up proj

        std_uk = (2 / (d_kC + latent_dim_kv)) ** 0.5
        nn.init.trunc_normal_(self.W_UK, mean=0.0, std=std_uk, a=-2.0 * std_uk, b=2.0 * std_uk)
        std_uv = (2 / (d_v + latent_dim_kv)) ** 0.5
        nn.init.trunc_normal_(self.W_UV, mean=0.0, std=std_uv, a=-2.0 * std_uv, b=2.0 * std_uv)

        self.out_proj = Linear(num_heads * d_v, d_model) #final o/p    
  
        self.rope = RoPE(d_kR, max_seq_len, theta) #rope for decoupled parts only

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:

        L_kv = self.W_DKV(x) #(batch_size, seq_len, latent_dim_kv)
        L_q = self.W_DQ(x) #(batch_size, seq_len, latent_dim_q)
        Q = self.W_UQ(L_q) #(b, t, num_heads*(d_kC + d_kR))
        '''
        einops.unpack requires the second argument to be a list of shapes (one shape per output tensor). 
        a shape must always be an iterable (like a list or tuple), even if it only contains a single dimension.

        [[size1], [size2]]: Passes two separate 1D shapes. 
        unpack cleanly outputs two distinct tensors with the exact dimensions required.
        '''
        [Q_C_packed, Q_R_packed] = unpack(Q, [[self.num_heads * self.d_kC], [self.num_heads * self.d_kR]], "b t *")
        Q_C = rearrange(Q_C_packed, "b t (h d_c) -> b h t d_c", h = self.num_heads)
        Q_R = rearrange(Q_R_packed, "b t (h d_r) -> b h t d_r", h = self.num_heads)

        Q_C_absorbed = einsum(Q_C, self.W_UK, 'b h t d, h d l -> b h t l')

        # (b, t, d_kR) -> (b 1 t d_kR)
        K_R = rearrange(self.W_KR(x), "b t d -> b 1 t d")

        scores_C = einsum(Q_C_absorbed, L_kv, 'b h t l, b s l -> b h t s')

        # pass through rope 
        Q_R = self.rope(Q_R, token_positions)
        K_R = self.rope(K_R, token_positions)
        K_R_expanded = repeat(K_R, "b 1 s d -> b h s d", h=self.num_heads)
        scores_R = einsum(Q_R, K_R_expanded, 'b h t d,b h s d -> b h t s ')

        total_score = scores_C + scores_R
        scale_factor = (self.d_kC + self.d_kR) ** -0.5
        scaled_scores = total_score * scale_factor

        seq_len = x.shape[-2]
        mask  = torch.tril(torch.ones(seq_len , seq_len , device=x.device , dtype=bool))
        # b h t s ( t = s during training) ; mask = False [ , , s , s]
        scaled_scores = scaled_scores.masked_fill(mask == False, float('-inf')) # PyTorch evaluates and broadcasts tensor dimensions from right to left.
        attn_weights = softmax(scaled_scores, dim=-1)  # ( b, h, t, s)

        output_latent = einsum(attn_weights, L_kv, 'b h t s, b s l -> b h t l')
        out = einsum(output_latent, self.W_UV, 'b h t l,h d l-> b h t d')
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(out)


class DecoderBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        latent_dim_q: int = 64,
        latent_dim_kv: int = 64,
        d_v: int = 32,
        d_kC: int = 32,
        d_kR: int = 32,
        max_seq_len: int = 512,
        theta: float = 10000.0,
        num_experts: int = 4, 
        top_k: int = 2
    ):
        super().__init__()
        from experts import MoE

        self.attn_norm = RMSNorm(d_model)
        self.attn = MLA(
            d_model=d_model,
            num_heads=num_heads,
            latent_dim_q=latent_dim_q,
            latent_dim_kv=latent_dim_kv,
            d_v=d_v,
            d_kC=d_kC,
            d_kR=d_kR,
            max_seq_len=max_seq_len,
            theta=theta
        )
        self.moe_norm = RMSNorm(d_model)
        self.moe = MoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        x = x + self.attn(self.attn_norm(x), token_positions)
        moe_out, aux_loss = self.moe(self.moe_norm(x))
        x = x + moe_out
        return x, aux_loss

        














        












        


