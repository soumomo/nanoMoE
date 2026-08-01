import torch
import torch.nn as nn

class AdamW(torch.optim.Optimizer):
    def __init__(self , params, lr:float = 1e-3 , betas: tuple[float , float] = (0.9 , 0.999) , eps: float = 1e-8 , weight_decay :float = 0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[1]}")
        
        if eps < 0:
            raise ValueError(f"Invaid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr = lr , betas = betas , eps = eps , weight_decay = weight_decay)
        super().__init__(params , defaults)


    def step(self , closure =None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue


                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                state = self.state[p]
                t = state.get("t" , 0)
                grad = p.grad.data


                if(len(state) == 0):
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                alpha_t = lr * ( (1.0 - beta2 ** t) ** 0.5 ) / (1.0 - beta1 ** t)

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-alpha_t)

        return loss





 



