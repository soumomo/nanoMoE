import sys
import os
import modal

app = modal.App("nanomoe-pipeline")

# Persistent volume to store data, tokenizer, checkpoints, and logs.
volume = modal.Volume.from_name("nanomoe-volume", create_if_missing=True)

# Container image: install all dependencies and embed nanoMoE source code.
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch~=2.11.0",
        "numpy>=2.4",
        "einops>=0.8",
        "einx>=0.4",
        "regex",
        "tqdm>=4.67",
        "gigatoken",
        "wandb>=0.25",
        "prettytable>=3.10.0",
        "safetensors",
        "huggingface_hub",
        "datasets>=3.0.0",
        "tiktoken",
    )
    .add_local_dir(
        "/Users/soumodeep/Resources/Projects/nanoMoE",
        remote_path="/root/nanoMoE",
        ignore=lambda path: any(
            part in [".venv", ".git", "__pycache__", "data", "checkpoints", "uv.lock"]
            for part in path.parts
        ) or str(path).endswith((".npy", ".png", ".pt", ".safetensors")),
    )
)

# W&B API key secret, created in the Modal dashboard as "wandb-secret".
wandb_secret = modal.Secret.from_name("wandb-secret")


# =====================================================================
# Train nanoMoE model on Modal H100 GPU (~70M params)
# =====================================================================
@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="H100",
    secrets=[wandb_secret],
    timeout=7200,
)
def train_model():
    """Launch a nanoMoE training run on a Modal H100 GPU container."""
    import subprocess

    os.makedirs("/vol/logs", exist_ok=True)
    os.makedirs("/vol/checkpoints", exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/nanoMoE"

    cmd = [
        sys.executable, "/root/nanoMoE/train.py",
        "--d_model", "512",
        "--num_heads", "8",
        "--num_layers", "8",
        "--d_ff", "1360",
        "--num_experts", "4",
        "--top_k", "2",
        "--context_length", "512",
        "--batch_size", "64",
        "--max_iters", "15000",
        "--max_lr", "3e-4",
        "--min_lr", "3e-5",
        "--warmup_iters", "750",
        "--weight_decay", "0.1",
        "--grad_clip_norm", "1.0",
        "--log_interval", "10",
        "--eval_interval", "500",
        "--checkpoint_interval", "1000",
        "--checkpoint_dir", "/vol/checkpoints",
        "--resume_from", "auto",
        "--log_file", "/vol/logs/train_log.csv",
        "--dataset_name", "HuggingFaceFW/fineweb-edu",
        "--dataset_subset", "sample-10BT",
        "--wandb",
        "--wandb_project", "nanoMoE",
        "--wandb_run_name", "nanoMoE-H100-FineWeb10BT",
    ]

    print("=" * 65)
    print("Starting nanoMoE training run on Modal H100 GPU")
    print("  Dataset: HuggingFaceFW/fineweb-edu (sample-10BT streaming)")
    print("  Model:   d_model=512, heads=8, layers=8, d_ff=1360, experts=4, top_k=2")
    print("  Target:  15,000 iterations (~491M tokens)")
    print("=" * 65)
    print("  Logging: /vol/logs/train_log.csv + WandB (project: nanoMoE)")
    print("=" * 65)

    subprocess.run(cmd, env=env, check=True)

    volume.commit()
    print("Training complete. Checkpoints and logs saved to /vol")


# =====================================================================
# Download logs & plot loss curves locally
# =====================================================================
@app.function(image=image, volumes={"/vol": volume}, timeout=120)
def fetch_logs():
    """Read the CSV log file from the volume and return its contents."""
    log_path = "/vol/logs/train_log.csv"
    if not os.path.exists(log_path):
        print("No log file found at /vol/logs/train_log.csv")
        return None

    with open(log_path, "r") as f:
        contents = f.read()

    print(f"Log file size: {len(contents)} bytes")
    return contents


def plot_losses(csv_text, save_path="loss_curves.png"):
    """Plot train/validation loss curves locally with a dark theme."""
    import csv
    from io import StringIO

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed locally. Install with: pip install matplotlib")
        with open("train_log.csv", "w") as f:
            f.write(csv_text)
        return

    reader = csv.DictReader(StringIO(csv_text))

    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    for row in reader:
        step = int(row["step"])
        if row.get("train_loss"):
            train_steps.append(step)
            train_losses.append(float(row["train_loss"]))
        if row.get("val_loss"):
            val_steps.append(step)
            val_losses.append(float(row["val_loss"]))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor="#09090B")
    ax.set_facecolor("#09090B")

    if train_losses:
        ax.plot(train_steps, train_losses, alpha=0.35, color="#CA8A04", linewidth=0.8, label="Train Loss (raw)")
        window = min(50, len(train_losses) // 5) if len(train_losses) > 10 else 1
        if window > 1:
            smoothed = []
            for i in range(len(train_losses)):
                start = max(0, i - window)
                smoothed.append(sum(train_losses[start:i + 1]) / (i - start + 1))
            ax.plot(train_steps, smoothed, color="#FACC15", linewidth=2.2, label="Train Loss (smoothed)")

    if val_losses:
        ax.plot(val_steps, val_losses, "o-", color="#F59E0B", linewidth=2.2, markersize=6, label="Val Loss")

    ax.set_xlabel("Step", fontsize=12, fontweight="bold", color="#FEF08A")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold", color="#FEF08A")
    ax.set_title("NANOMOE H100 TRAINING METRICS", fontsize=15, fontweight="bold", color="#FACC15", pad=15)

    ax.grid(True, linestyle="--", alpha=0.15, color="#FACC15")
    for spine in ax.spines.values():
        spine.set_color("#854D0E")
        spine.set_linewidth(1.2)

    ax.tick_params(colors="#EAB308", labelsize=10)
    legend = ax.legend(fontsize=10, facecolor="#18181B", edgecolor="#854D0E")
    for text in legend.get_texts():
        text.set_color("#FEF08A")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Loss curves saved to {save_path}")
    plt.close()


@app.function(image=image, volumes={"/vol": volume}, gpu="A10G", timeout=300)
def generate_text(prompt: str = "The future of AI is", max_tokens: int = 150):
    """Generate text from the trained nanoMoE model directly on Modal GPU."""
    import sys
    sys.path.append("/root/nanoMoE")
    from generate import load_nano_moe_model, generate
    
    ckpt_path = "/vol/checkpoints/checkpoint_final.pt"
    if not os.path.exists(ckpt_path):
        import glob
        ckpts = glob.glob("/vol/checkpoints/checkpoint_step_*.pt")
        if ckpts:
            ckpts.sort(key=lambda p: int(p.split("_")[-1].split(".")[0]))
            ckpt_path = ckpts[-1]
            
    print(f"Loading checkpoint from {ckpt_path} on Modal GPU...")
    model, enc, device = load_nano_moe_model(ckpt_path)
    return generate(model, enc, device, prompt, max_tokens=max_tokens)


@app.function(image=image, volumes={"/vol": volume}, timeout=300)
def export_safetensors(step: int = 14000):
    """Load PyTorch checkpoint from volume and save as clean model.safetensors."""
    import sys
    import glob
    import torch
    from safetensors.torch import save_file
    sys.path.append("/root/nanoMoE")
    
    ckpt_path = f"/vol/checkpoints/checkpoint_step_{step}.pt"
    if not os.path.exists(ckpt_path):
        ckpts = glob.glob("/vol/checkpoints/checkpoint_step_*.pt")
        if ckpts:
            ckpts.sort(key=lambda p: int(p.split("_")[-1].split(".")[0]))
            ckpt_path = ckpts[-1]
            
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    
    out_path = "/vol/model.safetensors"
    save_file(state_dict, out_path)
    volume.commit()
    print(f"Successfully exported clean {out_path} ({os.path.getsize(out_path) / (1024*1024):.2f} MB)")
    return out_path


@app.local_entrypoint()
def main(action: str = "train", prompt: str = "The future of artificial intelligence is", step: int = 14000):
    """Entry point for `modal run modal_pipeline.py --action [train|logs|generate|export] --prompt 'text'`."""
    if action == "train":
        print("Deploying and starting nanoMoE training run on Modal H100...")
        train_model.remote()
    elif action == "logs":
        print("Fetching logs from Modal Volume...")
        logs = fetch_logs.remote()
        if logs:
            plot_losses(logs)
    elif action == "generate":
        print(f"Generating output from trained model on Modal GPU for prompt: '{prompt}'...")
        res = generate_text.remote(prompt=prompt)
        print("\nGenerated Text Output:\n")
        print(res)
    elif action == "export":
        print(f"Exporting step {step} checkpoint to model.safetensors on Modal Volume...")
        res = export_safetensors.remote(step=step)
        print(f"Done! Download model.safetensors using:")
        print("  uv run modal volume get nanomoe-volume model.safetensors ./model.safetensors")
    else:
        print(f"Unknown action '{action}'. Usage: modal run modal_pipeline.py --action [train|logs|generate|export]")