import argparse
import sys
import os
import torch
import torch.nn.functional as F
import tiktoken

from model import TransformerLM

# ANSI styling: white text, background inherits whatever the terminal/IDE is already using
FG_WHITE = "\033[97m"
RESET = "\033[0m"
STYLE = FG_WHITE


def styled(text: str) -> str:
    return f"{STYLE}{text}{RESET}"


def load_nano_moe_model(checkpoint_path="checkpoints/checkpoint_step_14000.pt", device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    print("Building nanoMoE architecture...")
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=512,
        d_model=512,
        num_layers=8,
        num_heads=8,
        d_ff=1360,
        rope_theta=10000.0,
        num_experts=4,
        top_k=2
    )

    print(f"Loading checkpoint weights from {checkpoint_path}...")
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path, device="cpu")
        model.load_state_dict(state_dict)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready on device: {device} ({num_params:,} parameters)\n")
    return model, enc, device


def print_expert_table(model):
    """Print per-layer MoE expert utilization, styled and with a real balance check."""
    layer_rows = []
    avg_f = [0.0, 0.0, 0.0, 0.0]
    num_l = 0

    for i, layer in enumerate(model.layers):
        if hasattr(layer.moe, "last_f_i"):
            f = layer.moe.last_f_i.tolist()
            top_exp = f.index(max(f))
            layer_rows.append((i + 1, f, top_exp))
            for idx in range(4):
                avg_f[idx] += f[idx]
            num_l += 1

    if num_l == 0:
        return

    avg_f = [x / num_l for x in avg_f]
    uniform = 1.0 / len(avg_f)
    max_deviation = max(abs(x - uniform) for x in avg_f)
    # flag skew if any expert's average usage deviates more than 10 percentage points from uniform
    balance_label = "Balanced" if max_deviation < 0.10 else "Skewed"

    try:
        from prettytable import PrettyTable, TableStyle
        table = PrettyTable()
        table.set_style(TableStyle.SINGLE_BORDER)
        table.field_names = ["Layer", "Expert 0", "Expert 1", "Expert 2", "Expert 3", "Top Expert"]
        table.align["Layer"] = "l"

        for layer_num, f, top_exp in layer_rows:
            table.add_row([
                f"Layer {layer_num}",
                f"{f[0] * 100:.1f}%",
                f"{f[1] * 100:.1f}%",
                f"{f[2] * 100:.1f}%",
                f"{f[3] * 100:.1f}%",
                f"Expert {top_exp}",
            ])

        table.add_row([
            "Overall",
            f"{avg_f[0] * 100:.1f}%",
            f"{avg_f[1] * 100:.1f}%",
            f"{avg_f[2] * 100:.1f}%",
            f"{avg_f[3] * 100:.1f}%",
            balance_label,
        ])

        print(styled(str(table)))
        print()
    except ImportError:
        header = f"{'Layer':<9}{'Expert 0':<11}{'Expert 1':<11}{'Expert 2':<11}{'Expert 3':<11}"
        print(styled("EXPERT ROUTING METRICS (8 LAYERS)"))
        print(styled(header))
        print(styled("-" * len(header)))
        for layer_num, f, _ in layer_rows:
            row = f"{'Layer ' + str(layer_num):<9}{f[0]*100:>7.1f}%   {f[1]*100:>7.1f}%   {f[2]*100:>7.1f}%   {f[3]*100:>7.1f}%"
            print(styled(row))
        print(styled("-" * len(header)))
        overall = f"{'Overall':<9}{avg_f[0]*100:>7.1f}%   {avg_f[1]*100:>7.1f}%   {avg_f[2]*100:>7.1f}%   {avg_f[3]*100:>7.1f}%   ({balance_label})"
        print(styled(overall))
        print()


def generate(
    model,
    enc,
    device,
    prompt: str,
    max_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    context_length: int = 512
):
    tokens = enc.encode_ordinary(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    print(styled(f"\nPrompt: \"{prompt}\""))
    sys.stdout.write(f"{STYLE}Generation: {prompt}")
    sys.stdout.flush()
    printed_text = prompt

    with torch.no_grad():
        for _ in range(max_tokens):
            if x.size(1) > context_length:
                x_input = x[:, -context_length:]
            else:
                x_input = x

            logits, _, _ = model(x_input)
            logits = logits[0, -1, :]

            if repetition_penalty != 1.0:
                for token_id in set(x[0].tolist()):
                    if logits[token_id] < 0:
                        logits[token_id] *= repetition_penalty
                    else:
                        logits[token_id] /= repetition_penalty

            if temperature > 0:
                logits = logits / temperature

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs_shifted = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
                sorted_indices_to_remove = cumulative_probs_shifted > top_p
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, next_token_id.unsqueeze(0)), dim=1)
            current_text = enc.decode(x[0].tolist())
            delta = current_text[len(printed_text):]
            sys.stdout.write(delta)
            sys.stdout.flush()
            printed_text = current_text

    sys.stdout.write(RESET + "\n")
    sys.stdout.flush()
    print_expert_table(model)
    return printed_text


def main():
    parser = argparse.ArgumentParser(description="nanoMoE text generation")
    parser.add_argument("--checkpoint", type=str, default="model.safetensors", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt text (omit for interactive mode)")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling threshold")
    parser.add_argument("--max_tokens", type=int, default=150, help="Max tokens to generate per response")

    args = parser.parse_args()

    model, enc, device = load_nano_moe_model(args.checkpoint)

    if args.prompt:
        generate(model, enc, device, args.prompt, max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p)
    else:
        print(styled("NANOMOE INTERACTIVE PLAYGROUND"))
        print(styled("Type your prompt and press Enter. Type 'exit' or 'quit' to stop."))
        print()

        while True:
            try:
                prompt = input("Enter prompt > ").strip()
                if not prompt:
                    continue
                if prompt.lower() in ("exit", "quit"):
                    print("Goodbye.")
                    break
                generate(model, enc, device, prompt, max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                break


if __name__ == "__main__":
    main()