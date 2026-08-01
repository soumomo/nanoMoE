import argparse
import csv
import os
import torch
import torch.nn as nn
import numpy as np
from model import TransformerLM
from optimizer import AdamW
from training import get_batch, load_checkpoint, save_checkpoint
from nn_utils import lr_cosine_schedule, gradient_clipping
import time
from tokenizer import Tokenizer
import torch.nn.functional as F
from prettytable import PrettyTable


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a NanoMoE Transformer model."
    )

    # data paths and loading
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/train.npy",
        help="Path to training .npy file",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/validation.npy",
        help="Path to validation .npy file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hugging Face dataset name (e.g. HuggingFaceFW/fineweb-edu)",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default=None,
        help="Hugging Face dataset subset (e.g. sample-10BT)",
    )
    parser.add_argument(
        "--mmap",
        type=str,
        default="r",
        choices=["r", "r+", "w+", "c", "None"],
        help="Memory mapping mode",
    )

    # model architecture & MoE configs
    parser.add_argument(
        "--d_model", type=int, default=512, help="Embedding dimension size"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--rope_theta", type=float, default=10000.0, help="Base value for RoPE calculation"
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Number of transformer layers"
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=1360,
        help="Dimension of feed-forward network",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=4,
        help="Number of experts per MoE block",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Number of top experts selected per token",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=50257, help="Vocabulary size"
    )

    # training loop configs
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size per training step"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_iters", type=int, default=20000, help="Total training iterations"
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold",
    )

    # lr and scheduler
    parser.add_argument(
        "--max_lr", type=float, default=8e-4, help="Peak learning rate"
    )
    parser.add_argument(
        "--min_lr", type=float, default=8e-5, help="Minimum learning rate"
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=1000,
        help="Number of iterations for LR warmup",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay factor"
    )

    # evaluation and checkpointing
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="How often to run evaluation",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=50,
        help="Number of batches to run during evaluation",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="How often to save model checkpoints",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="How often to print training logs",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to CSV file for logging losses",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/vol/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="nanoMoE",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="auto",
        help="Checkpoint path to resume from ('auto' to search checkpoint_dir)",
    )

    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up data stream or memory-mapped numpy array
    if args.dataset_name:
        print(f"Streaming HuggingFace dataset '{args.dataset_name}' (subset: {args.dataset_subset})...")
        from datasets import load_dataset
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        args.vocab_size = enc.n_vocab

        ds = load_dataset(args.dataset_name, name=args.dataset_subset, split="train", streaming=True)

        def create_stream_generator(dataset_iter, enc, batch_size, context_length, device):
            buffer = []
            for sample in dataset_iter:
                text = sample.get("text", "")
                if not text:
                    continue
                tokens = enc.encode_ordinary(text)
                buffer.extend(tokens)

                required_tokens = batch_size * context_length + 1
                while len(buffer) >= required_tokens:
                    chunk = buffer[:required_tokens]
                    buffer = buffer[batch_size * context_length:]

                    t_tensor = torch.tensor(chunk, dtype=torch.long, device="cpu")
                    x = t_tensor[:-1].view(batch_size, context_length)
                    y = t_tensor[1:].view(batch_size, context_length)
                    yield x, y

        train_gen = create_stream_generator(ds, enc, args.batch_size, args.context_length, device)
        def get_train_batch():
            x_b, y_b = next(train_gen)
            return x_b.to(device), y_b.to(device)

        print("Pre-collecting validation batches for instant evaluation...")
        val_batches = []
        for _ in range(args.eval_iters):
            x_v, y_v = next(train_gen)
            val_batches.append((x_v, y_v))

        val_idx = [0]
        def get_val_batch():
            x_v, y_v = val_batches[val_idx[0] % len(val_batches)]
            val_idx[0] += 1
            return x_v.to(device), y_v.to(device)
    else:
        mmap_mode = None if args.mmap == "None" else args.mmap
        train_data = np.load(args.train_path, mmap_mode=mmap_mode)
        val_data = np.load(args.val_path, mmap_mode=mmap_mode)
        get_train_batch = lambda: get_batch(train_data, args.batch_size, args.context_length, device)
        get_val_batch = lambda: get_batch(val_data, args.batch_size, args.context_length, device)

    # Initialize NanoMoE Model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"NanoMoE initialized with {num_params:,} parameters ({args.num_experts} experts, top-{args.top_k} routing).")

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay
    )

    # Auto-resume checkpoint logic (robust to corruptions)
    start_step = 0
    resume_path = args.resume_from

    if resume_path == "auto":
        import glob
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpts = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_step_*.pt"))
        if ckpts:
            ckpts.sort(key=lambda p: int(p.split("_")[-1].split(".")[0]))
            for candidate in reversed(ckpts):
                try:
                    last_iteration = load_checkpoint(candidate, model, optimizer)
                    start_step = last_iteration + 1
                    print(f"Resumed successfully from {candidate}, continuing at step {start_step}")
                    resume_path = candidate
                    break
                except Exception as e:
                    print(f"Warning: Checkpoint {candidate} is corrupted ({e}). Removing and trying earlier checkpoint...")
                    try:
                        os.remove(candidate)
                    except OSError:
                        pass
            else:
                print("No valid uncorrupted checkpoint found. Starting fresh training run.")
                resume_path = None
        else:
            resume_path = None
    elif resume_path is not None and os.path.exists(resume_path):
        try:
            last_iteration = load_checkpoint(resume_path, model, optimizer)
            start_step = last_iteration + 1
            print(f"Resumed from {resume_path}, continuing at step {start_step}")
        except Exception as e:
            print(f"Warning: Specified checkpoint {resume_path} failed to load ({e}). Starting fresh training run.")
            start_step = 0
    else:
        print("Starting fresh training run.")

    model.train()
    start_time = time.time()
    best_val_loss = float("inf")

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb_id = None
        wandb_id_file = os.path.join(args.checkpoint_dir, "wandb_run_id.txt")
        if start_step > 0 and os.path.exists(wandb_id_file):
            with open(wandb_id_file, "r") as f:
                wandb_id = f.read().strip()

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=wandb_id,
            resume="allow" if wandb_id else None,
            config={
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "d_ff": args.d_ff,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
                "vocab_size": args.vocab_size,
                "context_length": args.context_length,
                "batch_size": args.batch_size,
                "max_iters": args.max_iters,
                "max_lr": args.max_lr,
                "min_lr": args.min_lr,
                "warmup_iters": args.warmup_iters,
                "weight_decay": args.weight_decay,
                "grad_clip_norm": args.grad_clip_norm,
                "num_params": num_params,
            },
        )
        if wandb.run and not wandb_id:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            with open(wandb_id_file, "w") as f:
                f.write(wandb.run.id)

    csv_file = None
    csv_writer = None
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        file_exists = os.path.exists(args.log_file) and start_step > 0
        csv_file = open(args.log_file, "a" if file_exists else "w", newline="")
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            csv_writer.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])

    for step in range(start_step, args.max_iters):
        lr = lr_cosine_schedule(
            step, args.max_lr, args.min_lr, args.warmup_iters, args.max_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_train_batch()

        use_amp = device.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits, loss, aux_loss = model(x, y)
        loss.backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        optimizer.step()

        if step % args.log_interval == 0 or step == args.max_iters - 1:
            elapsed = time.time() - start_time
            train_loss_val = loss.item()
            aux_loss_val = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

            print(
                f"step {step:5d} | loss: {train_loss_val:6.4f} (aux: {aux_loss_val:6.4f}) | lr: {lr:.2e} | time: {elapsed:.1f}s"
            )
            if csv_writer:
                csv_writer.writerow([step, f"{train_loss_val:.6f}", "", f"{lr:.2e}", f"{elapsed:.1f}"])
                csv_file.flush()
            if use_wandb:
                wandb.log({
                    "train/loss": train_loss_val,
                    "train/aux_loss": aux_loss_val,
                    "train/lr": lr,
                    "train/step": step,
                    "train/elapsed_s": elapsed
                }, step=step)

        if step % args.eval_interval == 0 and step > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(args.eval_iters):
                    x_val, y_val = get_val_batch()
                    _, loss_val, _ = model(x_val, y_val)
                    val_losses.append(loss_val.item())

            val_loss = np.mean(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            print(f"\nEvaluation at step {step} | Val Loss: {val_loss:6.4f} (Best: {best_val_loss:6.4f})\n")
            if csv_writer:
                csv_writer.writerow([step, "", f"{val_loss:.6f}", f"{lr:.2e}", f"{time.time() - start_time:.1f}"])
                csv_file.flush()
            if use_wandb:
                wandb.log({"val/loss": val_loss, "train/step": step}, step=step)

            model.train()

        if step % args.checkpoint_interval == 0 and step > 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{step}.pt")
            save_checkpoint(model, optimizer, step, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"Saved final checkpoint to {final_path}")

    if csv_file:
        csv_file.close()


if __name__ == "__main__":
    main()