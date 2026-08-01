import os
import torch 
import numpy as np
from tokenizer import train_bpe_tokenizer, Tokenizer

def encode_file_fast(tokenizer, file_path, max_lines=None, batch_size=50000):
    """Encodes text file in fast parallel batches using gigatoken."""
    all_tokens = []
    batch = []
    lines_processed = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            batch.append(line)
            lines_processed += 1

            if max_lines and lines_processed >= max_lines:
                break

            if len(batch) >= batch_size:
                batch_encoded = tokenizer.encode_batch(batch)
                for ids in batch_encoded:
                    all_tokens.extend(ids)
                batch = []
                print(f"  Processed {lines_processed:,} lines...")

        if batch:
            batch_encoded = tokenizer.encode_batch(batch)
            for ids in batch_encoded:
                all_tokens.extend(ids)
            print(f"  Processed {lines_processed:,} lines (total)...")

    return np.array(all_tokens, dtype=np.uint16)


if __name__ == "__main__":
    train_text_path = "/vol/TinyStoriesV2-GPT4-train.txt"
    valid_text_path = "/vol/TinyStoriesV2-GPT4-valid.txt"

    print("=" * 60)
    print("Step 2a: Training BPE Tokenizer with gigatoken...")
    print("=" * 60)

    tokenized_data = {}
    tokenized_data["vocab"], tokenized_data["merges"] = train_bpe_tokenizer(
        train_text_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )

    os.makedirs("/vol", exist_ok=True)
    torch.save(tokenized_data, "/vol/tokenizer.pt")
    print("Tokenizer saved to /vol/tokenizer.pt")

    tokenizer = Tokenizer(
        vocab=tokenized_data["vocab"],
        merges=tokenized_data["merges"],
        special_tokens=["<|endoftext|>"]
    )

    # -----------------------------------------------------------------
    # Process Train Data (FULL dataset, no line limit)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2b: Tokenizing Full Training Dataset...")
    print("=" * 60)
    train_array = encode_file_fast(tokenizer, train_text_path, max_lines=None)
    np.save("/vol/train.npy", train_array)
    print(f"Saved /vol/train.npy: {len(train_array):,} tokens ({train_array.nbytes / (1024**2):.1f} MB)")
    del train_array

    # -----------------------------------------------------------------
    # Process Validation Data
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2c: Tokenizing Validation Dataset...")
    print("=" * 60)
    valid_array = encode_file_fast(tokenizer, valid_text_path)
    np.save("/vol/validation.npy", valid_array)
    print(f"Saved /vol/validation.npy: {len(valid_array):,} tokens ({valid_array.nbytes / (1024**2):.1f} MB)")
    del valid_array

    print("\nData preparation finished successfully!")