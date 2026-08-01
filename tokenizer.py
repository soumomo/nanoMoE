# =====================================================================
# OLD: pure-python pretokenization regex (no longer needed with gigatoken)
# =====================================================================
# PRE_TOKEN_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# import regex as re 
# from multiprocessing import Pool

import os
import json
import gigatoken


# =====================================================================
# OLD: pretokenize_chunk - pure-python parallel pretokenization
# (replaced by gigatoken.train_bpe which handles this in Rust)
# =====================================================================
# def pretokenize_chunk(chunk , special_tokens):
#     '''process one chunk - runs in parallel'''
#     word_frequencies = {}
#
#     # split on special tokens so they never get merged into pretokenization
#     if special_tokens:
#         sorted_specials = sorted(special_tokens, key=len, reverse=True)
#         pattern = "|".join(re.escape(tok) for tok in sorted_specials)
#         segments = re.split(f"({pattern})", chunk)
#     else:
#         segments = [chunk]
#
#     for segment in segments:
#         if not segment:
#             continue
#         if segment in special_tokens:
#             # special tokens never participate in pretokenization/merges
#             continue
#         for match in re.finditer(PRE_TOKEN_REGEX , segment):
#             word = match.group()
#             word_bytes  = tuple(word.encode("utf-8"))
#             word_frequencies[word_bytes] = word_frequencies.get(word_bytes , 0) + 1
#     return word_frequencies

# =====================================================================
# OLD: find_chunk_boundaries - split corpus into chunks at document boundaries
# (replaced by gigatoken.train_bpe which handles chunking in Rust)
# =====================================================================
# def find_chunk_boundaries(file , num_chunks , boundary_token):
#     file_size = file.seek(0 , os.SEEK_END)
#     num_chunks = max(1, min(num_chunks, file_size))  # never more chunks than bytes
#     chunk_size = file_size//num_chunks
#     boundaries = [i * chunk_size for i in range(num_chunks + 1)]
#     boundaries[-1] = file_size  # guarantee correctness regardless of rounding
#
#     
#     #adjust vboundaries to lanf on special tokens (document boundaries)
#     for i in range(1 , len(boundaries) - 1):
#         file.seek(boundaries[i])
#         offset = 0
#
#         # search for next occurence of boundary token
#         while True:
#             chunk = file.read(4096)
#             if not chunk:
#                 boundaries[i] = file_size
#                 break
#
#             pos = chunk.find(boundary_token)
#             if pos != -1:
#                 boundaries[i] += offset + pos
#                 break
#             offset += len(chunk)
#
#     boundaries = sorted(set(boundaries))
#     return boundaries


# =====================================================================
# OLD: train_bpe_tokenizer - pure-python BPE training loop
# (replaced by gigatoken.train_bpe below)
# =====================================================================
# def train_bpe_tokenizer(input_path , vocab_size , special_tokens , num_processes = None):
#     if num_processes is None:
#         num_processes = os.cpu_count()
#
#         #split corpus into chunks
#     with open(input_path , "rb") as f:
#         boundaries = find_chunk_boundaries(f , num_processes*3 , b"<|endoftext|>")
#         chunks = []
#         for start , end in zip(boundaries[:-1] , boundaries[1:]):
#             f.seek(start)
#             chunks.append(f.read(end-start).decode("utf-8"))
#
#     # parallel pre-tokenization
#     with Pool(num_processes) as pool:
#         chunk_frequencies = pool.starmap(
#             pretokenize_chunk,
#             [(chunk , special_tokens) for chunk in chunks]
#
#         )
#
#     #combine word frequncies from al chunks
#     word_frequencies = {}
#
#     for chunk_freq in chunk_frequencies:
#         for word, freq in chunk_freq.items():
#             word_frequencies[word] = (
#                 word_frequencies.get(word, 0) + freq
#             )
#
#
#     #initialize vocabulary with all byte values
#     vocab = {idx: bytes([idx]) for idx in range(256)}
#
#     #add special tokens
#     for i , token in enumerate(special_tokens):
#         vocab[256+i] = token.encode("utf-8")
#     
#     merges = []
#     num_merges = vocab_size - len(vocab)
#     #count all the adjacent pairs across all words once
#     pair_frequencies = {}
#     
#     def get_pairs(word):
#         return [(word[i] , word[i+1]) for i in range(len(word) - 1)]
#     
#     for word , freq in word_frequencies.items():
#         for pair in get_pairs(word):
#             pair_frequencies[pair] = pair_frequencies.get(pair , 0) + freq
#
# # Now the merge loop only updates what changes:
#
#     for _ in range(num_merges):
#         if not pair_frequencies:
#             break
#             
#         # find the most frequent pair (tie-breakoing lexicographically)
#
#         best_pair = max(
#             pair_frequencies.keys() , 
#             key = lambda p: (pair_frequencies[p] , vocab[p[0]] , vocab[p[1]])
#         )
#             
#         #create new token
#         new_id = len(vocab)
#         vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
#         merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
#
#         # apply merge to all words
#         new_word_frequencies = {}
#
#         for word , freq in word_frequencies.items():
#             if best_pair not in get_pairs(word):
#                 #word is unchanged so just copy it
#                 new_word_frequencies[word] = freq
#             else:
#                 #subtract old pair counts
#                 for pair in get_pairs(word):
#                     pair_frequencies[pair] -= freq
#                     if pair_frequencies[pair] == 0:
#                         del pair_frequencies[pair]
#                 
#                 #apply the merge
#                 new_word = []
#                 i = 0
#                 while i < len(word):
#                     if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
#                         new_word.append(new_id)
#                         i += 2
#                     else:
#                         new_word.append(word[i])
#                         i += 1
#                 new_word_frequencies[tuple(new_word)] = freq
#
#                 for pair in get_pairs(new_word):
#                     pair_frequencies[pair] = pair_frequencies.get(pair , 0) + freq
#
#
#         word_frequencies = new_word_frequencies
#
#     return vocab , merges


# =====================================================================
# NEW: gigatoken-powered train_bpe_tokenizer
# Uses Rust multithreading for pretokenization + BPE merge loop
# Same function signature so data_prep.py and train.py don't need changes
# =====================================================================
def train_bpe_tokenizer(input_path, vocab_size, special_tokens, num_processes=None):
    """
    Trains a BPE tokenizer using gigatoken's Rust engine.
    Accepts a file path (str) or raw bytes. num_processes is accepted
    for backwards compatibility but ignored (gigatoken manages its own threads).
    Returns (vocab, merges) in the same format as the old pure-python version.
    """
    if isinstance(input_path, str) and os.path.exists(input_path):
        in_data = input_path
    elif isinstance(input_path, bytes):
        in_data = input_path
    else:
        in_data = str(input_path).encode("utf-8")

    vocab, merges = gigatoken.train_bpe(
        in_data=in_data,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    return vocab, merges


# =====================================================================
# Helper: convert (vocab, merges) to HuggingFace tokenizer.json format
# Needed to bridge gigatoken.train_bpe output -> gigatoken.Tokenizer
# =====================================================================
def _bytes_to_unicode():
    """HuggingFace GPT-2 ByteLevel byte->unicode mapping."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(x) for x in cs]))


def _export_to_tokenizer_json(vocab, merges, special_tokens=None):
    """Convert (vocab, merges) byte dicts to HuggingFace tokenizer.json string."""
    byte_encoder = _bytes_to_unicode()

    def b_to_str(b_seq):
        return ''.join(byte_encoder[b] for b in b_seq)

    specials_set = set(t.encode('utf-8') for t in (special_tokens or []))
    hf_vocab = {}
    added_tokens = []

    for idx, b_seq in vocab.items():
        if b_seq in specials_set:
            tok_str = b_seq.decode('utf-8', errors='replace')
            hf_vocab[tok_str] = idx
            added_tokens.append({
                "id": idx, "content": tok_str, "single_word": False,
                "lstrip": False, "rstrip": False, "normalized": False, "special": True
            })
        else:
            hf_vocab[b_to_str(b_seq)] = idx

    hf_merges = [f"{b_to_str(p0)} {b_to_str(p1)}" for p0, p1 in merges]

    tokenizer_json = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": added_tokens, "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None,
            "continuing_subword_prefix": None, "end_of_word_suffix": None,
            "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
            "vocab": hf_vocab, "merges": hf_merges
        }
    }
    return json.dumps(tokenizer_json)


# =====================================================================
# OLD: Tokenizer class - pure-python encode/decode
# (replaced by gigatoken-backed Tokenizer below)
# =====================================================================
# class Tokenizer:
#     def __init__(self , vocab: dict ,merges: list , special_tokens: list = None):
#          self.decoder = vocab # {idx: bytes([idx]) for idx in range(256)}
#         # reverses {idx: bytes} into {bytes: idx}
#          self.encoder = {bytes_val :idx for idx , bytes_val in vocab.items()}
#          self.merges = merges
#          
#          self.special_tokens = special_tokens or []
#
#          if self.special_tokens:
#             sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
#             pattern = "|".join(re.escape(tok) for tok in sorted_specials)
#             self.special_pattern = re.compile(f"({pattern})")
#          else:
#             self.special_pattern = None
#              
#
#
#     def decode(self , ids: list[int]) -> str:
#         byte_chunks = [self.decoder[idx] for idx in ids]
#         all_bytes = b"".join(byte_chunks)
#         return all_bytes.decode("utf-8" , errors ="replace")    
#
#     def encode(self , text: str) -> list[int]:
#         if self.special_tokens:
#             segments = self.special_pattern.split(text)
#         else:
#             segments = [text]
#
#         final_ids = []
#         
#         for segment in segments:
#             if not segment:
#                 continue
#                 
#             # check if this specific segment is a special token
#             if segment in self.special_tokens:
#                 final_ids.append(self.encoder[segment.encode("utf-8")])
#                 continue  
#             
#             # If it is not a special token, process it as normal text
#             for match in re.finditer(PRE_TOKEN_REGEX, segment):
#                 word = match.group()
#                 word_ids = [self.encoder[bytes([b])] for b in word.encode("utf-8")]
#
#             
#                 for p0 , p1 in self.merges:
#                     combined_bytes = p0 + p1
#                     new_id = self.encoder[combined_bytes]
#
#                     id0 = self.encoder[p0]
#                     id1 = self.encoder[p1]
#                     pair_to_find = (id0 , id1)
#
#
#                     #scan and replace the pair with new ID
#                     new_word = []
#                     i = 0
#                     while i < len(word_ids):
#                         if i < len(word_ids) - 1 and (word_ids[i], word_ids[i + 1]) == pair_to_find:
#                             new_word.append(new_id)
#                             i += 2
#                         else:
#                             new_word.append(word_ids[i])
#                             i += 1
#                     word_ids = new_word
#
#                 final_ids.extend(word_ids)
#
#
#         return final_ids
#                 
#
#
#
#     def encode_iterable(self, texts):
#         """
#         > accepts an iterable of strings (like lines in a file).
#         > yields token IDs one by one dynamically (as a Python generator)
#         > instead of storing the whole file's tokens in memory all at once.
#         """
#         for text in texts:
#             ids = self.encode(text)
#             for token_id in ids:
#                 yield token_id


# =====================================================================
# NEW: gigatoken-backed Tokenizer class
# Same interface (vocab, merges, special_tokens) so train.py works unchanged.
# Keeps .encoder and .decoder attributes for backwards compat (e.g. eos_id lookup).
# =====================================================================
class Tokenizer:
    def __init__(self, vocab: dict = None, merges: list = None, special_tokens: list = None):
        self.special_tokens = special_tokens or []

        if vocab is not None and merges is not None:
            self.decoder = vocab
            self.encoder = {bytes_val: idx for idx, bytes_val in vocab.items()}
            self.merges = merges
            json_str = _export_to_tokenizer_json(vocab, merges, self.special_tokens)
            self._gt_tokenizer = gigatoken.Tokenizer.from_json(json_str)
        else:
            raise ValueError("Tokenizer must be initialized with vocab and merges.")

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to a string."""
        decoded_bytes = self._gt_tokenizer.decode(ids)
        if isinstance(decoded_bytes, bytes):
            return decoded_bytes.decode("utf-8", errors="replace")
        return str(decoded_bytes)

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        encoded = self._gt_tokenizer.encode(text)
        return [int(tok_id) for tok_id in encoded]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a list of text strings into lists of token IDs in parallel."""
        return self._gt_tokenizer.encode_batch_list(texts)

    def encode_files(self, source):
        """Directly encode raw text files in parallel using Rust memory mapping."""
        return self._gt_tokenizer.encode_files(source)

    def encode_iterable(self, texts):
        """
        > accepts an iterable of strings (like lines in a file).
        > yields token IDs one by one dynamically (as a Python generator)
        > instead of storing the whole file's tokens in memory all at once.
        """
        for text in texts:
            for token_id in self.encode(text):
                yield token_id

    def save_json(self, save_path: str):
        """Save the tokenizer in standard HuggingFace tokenizer.json format."""
        json_str = _export_to_tokenizer_json(self.decoder, self.merges, self.special_tokens)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json_str)
