import torch
import torch.nn.functional as F
import torch.nn as nn   
from model import Linear, SwiGLU, softmax
from einops import repeat, rearrange

class MLPExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.ffn = SwiGLU(d_model, d_ff)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MoE(nn.Module):
    def __init__(self, d_model: int = 128, d_ff: int = 256, num_experts: int = 4, bias: bool = False, dropout: float = 0.1, top_k: int = 2,use_noisy_top_k: bool = True):
        # left the bias and dropout arguments for any future updates (if so (^^'))
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_noisy_top_k = use_noisy_top_k
    
        self.experts = nn.ModuleList([MLPExpert(d_model, d_ff) for _ in range(num_experts)])
        self.router = Linear(d_model, num_experts)
        if self.use_noisy_top_k:
            self.w_noise = Linear(d_model, num_experts)
            nn.init.normal_(self.w_noise.weight, std=1e-3)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_reshaped = rearrange(x, 'b t d -> (b t) d')
        

        clean_logits = self.router(x_reshaped) # [num_tokens, num_experts]
        if self.training and self.use_noisy_top_k:
            noise_std = F.softplus(self.w_noise(x_reshaped))
            noise = noise_std * torch.randn_like(noise_std)
            router_logits = clean_logits + noise
        else:
            router_logits  = clean_logits
            
        router_probs = softmax(router_logits, dim = -1)

        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim = -1)
        normalized_topk_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # implement the aux loss 
        mask = torch.zeros_like(router_probs)
        mask.scatter_(1,top_k_indices,1.0) # Out-of-place version of torch.Tensor.scatter_()
        f_i = mask.mean(dim = 0)
        P_i = router_probs.mean(dim = 0)

        # added for checking the expert probs
        self.last_f_i = f_i.detach()

        aux_loss = self.num_experts * torch.sum(f_i * P_i)

        out = torch.zeros_like(x_reshaped)
        for i in range(self.num_experts):
            token_idx, expert_idx = torch.where(top_k_indices == i)
            if token_idx.numel() == 0:
                continue
            expert_input = x_reshaped[token_idx]
            expert_output = self.experts[i](expert_input)
            weight = normalized_topk_weights[token_idx, expert_idx]
            out[token_idx] += expert_output * weight.unsqueeze(-1) #batch_size -> [batch_size, 1]

        out = rearrange(out, '(b t) d -> b t d', b=batch_size, t=seq_len)
        return out, aux_loss


if __name__ == "__main__":
    ip = torch.rand(2,8,128)
    moe = MoE()
    op, aux_loss = moe(ip)

    print(f"Input shape:     {list(ip.shape)}")
    print(f"Output shape:    {list(op.shape)}")
    print(f"Auxiliary loss:  {aux_loss.item():.4f}")

    assert op.shape == ip.shape, f"Shape mismatch: {op.shape} vs {ip.shape}"
    print("thrilled to announce that my MoE is working!")




            


            




