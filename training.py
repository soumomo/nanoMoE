import torch
import torch.nn as nn
import numpy as np

def get_batch(dataset, batch_size, context_length, device):
    rng = np.random.default_rng()
    upper_bound = len(dataset) - context_length
    start_indices = rng.integers(0 , upper_bound , batch_size)

    # shape: (batch_size, 1) + (context_length,) -> (batch_size, context_length)
    grid_indices = start_indices[:, None] + np.arange(context_length) #broadcasting 
    x_np = dataset[grid_indices]
    y_np = dataset[grid_indices + 1]
    x = torch.from_numpy(x_np).to(device, dtype=torch.long)
    y = torch.from_numpy(y_np).to(device, dtype=torch.long)
    return (x , y)



def save_checkpoint(model, optimizer, iteration, out):
    import os
    from safetensors.torch import save_file

    safetensors_out = out.replace(".pt", ".safetensors")
    state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, safetensors_out)

    checkpoint = {
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    tmp_out = f"{out}.tmp"
    torch.save(checkpoint, tmp_out)
    os.replace(tmp_out, out)


def load_checkpoint(src, model, optimizer=None):
    import os
    from safetensors.torch import load_file

    safetensors_src = src.replace(".pt", ".safetensors")
    if os.path.exists(safetensors_src):
        state_dict = load_file(safetensors_src)
        model.load_state_dict(state_dict)
    
    iteration = 0
    if os.path.exists(src):
        checkpoint = torch.load(src, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "model" in checkpoint and not os.path.exists(safetensors_src):
                model.load_state_dict(checkpoint["model"])
            if optimizer is not None and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            iteration = checkpoint.get("iteration", 0)

    return iteration 






    

