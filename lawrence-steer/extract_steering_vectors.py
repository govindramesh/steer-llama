#!/usr/bin/env python3
"""
Extract steering vectors from Pi vs Llama activations.
Computes mean difference vectors across samples and tokens for each layer.

Example usage:
python extract_steering_vectors.py \
    --h5-path /mnt/vast/home/lawrence/steer-llama/outputs/activations/activations_emoji.h5 \
    --output /mnt/vast/home/lawrence/steer-llama/outputs/steering_vectors/steering_vectors_emoji.pt \
    --token-strategy all
"""

import argparse
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import re

# Added for emoji token strategy
from transformers import AutoTokenizer
import emoji


def discover_sample_indices(h5_path: str) -> List[int]:
    """Discover available sample indices by scanning the 'activations' group.
    Returns sorted unique indices for which both chosen and rejected groups exist.
    """
    pattern = re.compile(r"^sample_(\d+)_(chosen|rejected)$")
    with h5py.File(h5_path, 'r') as f:
        if 'activations' not in f:
            raise ValueError("HDF5 file missing 'activations' group")
        keys = list(f['activations'].keys())
    index_to_kinds: Dict[int, set] = {}
    for k in keys:
        m = pattern.match(k)
        if not m:
            continue
        idx = int(m.group(1))
        kind = m.group(2)
        index_to_kinds.setdefault(idx, set()).add(kind)
    indices = sorted([i for i, kinds in index_to_kinds.items() if {'chosen', 'rejected'}.issubset(kinds)])
    if not indices:
        raise ValueError("No complete (chosen,rejected) samples found in activations group")
    return indices


def load_activations(h5_path: str, sample_indices: List[int]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Load activations for specified samples.
    
    Returns:
        pi_activations: Dict mapping layer_name -> List[tensor] (each tensor is (1, seq_len, hidden_dim))
        llama_activations: Same format
    """
    pi_activations = {}
    llama_activations = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Get layer names from first sample
        first_chosen_key = f"sample_{sample_indices[0]}_chosen"
        layer_names = sorted(f['activations'][first_chosen_key].keys())
        
        print(f"Found {len(layer_names)} layers")
        print(f"Loading activations for {len(sample_indices)} samples...")
        
        # Initialize lists for each layer
        for layer_name in layer_names:
            pi_tensors = []
            llama_tensors = []
            
            for sample_idx in sample_indices:
                chosen_key = f"sample_{sample_idx}_chosen"
                rejected_key = f"sample_{sample_idx}_rejected"
                
                # Load Pi (chosen) activations
                pi_activation = torch.tensor(f['activations'][chosen_key][layer_name][:])
                pi_tensors.append(pi_activation)
                
                # Load Llama (rejected) activations
                llama_activation = torch.tensor(f['activations'][rejected_key][layer_name][:])
                llama_tensors.append(llama_activation)
            
            # Store as lists (don't stack due to variable sequence lengths)
            pi_activations[layer_name] = pi_tensors
            llama_activations[layer_name] = llama_tensors
    
    return pi_activations, llama_activations


def _find_assistant_response_start(tokenizer: AutoTokenizer, tokens: torch.Tensor) -> int:
    """Find the index where the assistant response begins, following chat template markers."""
    pattern = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)

    seq_list = tokens[0].tolist()  # remove batch dimension
    m = len(pattern)

    match_indices: List[int] = []
    for j in range(0, len(seq_list) - m + 1):
        if seq_list[j:j + m] == pattern:
            match_indices.append(j)
    if not match_indices:
        raise ValueError("Assistant header pattern '<|start_header_id|>assistant<|end_header_id|>' not found in tokenized conversation.")

    # If there is a trailing generation prompt header, select the penultimate occurrence
    header_idx = match_indices[-2] if len(match_indices) >= 2 else match_indices[-1]

    start_idx = header_idx + m
    if newline_ids and len(newline_ids) == 1 and start_idx < len(seq_list) and seq_list[start_idx] == newline_ids[0]:
        start_idx += 1
    return start_idx


def _build_emoji_mask_for_response(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    response_text: str,
) -> torch.Tensor:
    """
    Build a boolean mask over the assistant response tokens indicating which tokens contain emoji.

    Returns a 1D boolean tensor of length equal to number of tokens from assistant response start to end.
    """
    # Tokenize response text alone to align mask directly with response tokens
    tokenized = tokenizer(
        response_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors=None,
    )

    offsets = tokenized.get("offset_mapping")
    input_ids = tokenized["input_ids"]

    mask_list: List[bool] = []
    if offsets is None:
        # Fallback: per-token decode (less reliable for emojis split across tokens)
        for tok_id in input_ids:
            piece = tokenizer.decode([tok_id], skip_special_tokens=False)
            mask_list.append(bool(emoji.emoji_list(piece)))
    else:
        # HF fast tokenizers return a list of (start, end) pairs per token
        for (start, end), tok_id in zip(offsets, input_ids):
            if start is None or end is None:
                mask_list.append(False)
                continue
            if end <= start:
                mask_list.append(False)
                continue
            substr = response_text[start:end]
            mask_list.append(bool(emoji.emoji_list(substr)))

    return torch.tensor(mask_list, dtype=torch.bool)


# New: Markdown detection helpers

def _find_markdown_marker_char_mask(text: str) -> List[bool]:
    """Return a per-character boolean mask for simple markdown markers in the text.
    Markers considered:
      - '**' (both asterisks as a unit)
      - '*' (single asterisk not part of '**')
      - '_' (underscore)
      - '#' (hash; each occurrence counts)
      - numbered list markers: digits followed by period or parenthesis
    We match anywhere in the text (not restricted to line starts) and do not special-case code blocks.
    """
    n = len(text)
    char_mask = [False] * n

    # 1) Mark all '**' pairs first (as single units)
    for m in re.finditer(r"\*\*", text):
        for i in range(m.start(), m.end()):
            char_mask[i] = True

    # 2) Single '*' not already covered by '**'
    for m in re.finditer(r"\*", text):
        s, e = m.start(), m.end()
        if not any(char_mask[s:e]):
            for i in range(s, e):
                char_mask[i] = True

    # 3) Underscores '_'
    for m in re.finditer(r"_", text):
        for i in range(m.start(), m.end()):
            char_mask[i] = True

    # 4) Hashes '#'
    for m in re.finditer(r"#", text):
        for i in range(m.start(), m.end()):
            char_mask[i] = True

    # 5) Numbered list markers: digits followed by period or parenthesis
    for m in re.finditer(r"\d+[.)]", text):
        for i in range(m.start(), m.end()):
            char_mask[i] = True

    return char_mask


def _find_formatting_char_mask(text: str) -> List[bool]:
    """Return a per-character boolean mask for formatting/special characters.
    Includes: spaces, newlines, punctuation, symbols, digits, etc.
    Excludes: letters only.
    """
    n = len(text)
    char_mask = [False] * n
    
    for i, char in enumerate(text):
        # Mark character if it's not a letter (includes spaces, punctuation, digits, symbols)
        if not char.isalpha():
            char_mask[i] = True
    
    return char_mask


def _build_markdown_mask_for_response(
    tokenizer: AutoTokenizer,
    response_text: str,
) -> torch.Tensor:
    """Build a boolean mask over response tokens that contain any markdown marker characters.
    Uses offsets mapping to map regex-matched character positions to tokens.
    """
    tokenized = tokenizer(
        response_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    offsets = tokenized.get("offset_mapping")
    char_mask = _find_markdown_marker_char_mask(response_text)

    token_mask: List[bool] = []
    if offsets is None:
        # Fallback: mark token if its decoded piece contains any marker characters
        ids = tokenized["input_ids"]
        for tok_id in ids:
            piece = tokenizer.decode([tok_id], skip_special_tokens=False)
            token_mask.append(bool(re.search(r"\*|_|#|\d+[.)]", piece)))
    else:
        for (start, end) in offsets:
            if start is None or end is None or end <= start:
                token_mask.append(False)
                continue
            has_marker = any(char_mask[start:end])
            token_mask.append(bool(has_marker))

    return torch.tensor(token_mask, dtype=torch.bool)


def _build_formatting_mask_for_response(
    tokenizer: AutoTokenizer,
    response_text: str,
) -> torch.Tensor:
    """Build a boolean mask over response tokens that contain any formatting/special characters.
    Uses offsets mapping to map character positions to tokens.
    """
    tokenized = tokenizer(
        response_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    offsets = tokenized.get("offset_mapping")
    char_mask = _find_formatting_char_mask(response_text)

    token_mask: List[bool] = []
    if offsets is None:
        # Fallback: mark token if its decoded piece contains any non-letter characters
        ids = tokenized["input_ids"]
        for tok_id in ids:
            piece = tokenizer.decode([tok_id], skip_special_tokens=False)
            token_mask.append(any(not c.isalpha() for c in piece))
    else:
        for (start, end) in offsets:
            if start is None or end is None or end <= start:
                token_mask.append(False)
                continue
            has_formatting = any(char_mask[start:end])
            token_mask.append(bool(has_formatting))

    return torch.tensor(token_mask, dtype=torch.bool)


def _load_dataset_pairs(dataset_json_path: str) -> List[Tuple[str, str, str]]:
    """
    Load ordered list of (prompt, pi_response, llama_response) from dataset JSON.
    Order must match the JSON insertion order, which is preserved in Python 3.7+.
    """
    with open(dataset_json_path, 'r') as f:
        data = json.load(f)
    pairs: List[Tuple[str, str, str]] = []
    for prompt_text, responses in data.items():
        pi_resp = responses.get("pi_response")
        llama_resp = responses.get("llama_response")
        if isinstance(pi_resp, str) and isinstance(llama_resp, str):
            pairs.append((prompt_text, pi_resp, llama_resp))
    return pairs


def compute_steering_vectors(
    pi_activations: Dict[str, List[torch.Tensor]],
    llama_activations: Dict[str, List[torch.Tensor]],
    sample_weights: Optional[List[float]] = None,
    token_strategy: str = "all",
    emoji_masks: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    markdown_masks: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    formatting_masks: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute steering vectors as Pi_mean - Llama_mean for each layer.
    
    Args:
        pi_activations: Dict mapping layer -> List[tensor] (each tensor is (1, seq_len, hidden_dim))
        llama_activations: Same format
        sample_weights: Optional weights for each sample (defaults to uniform)
        token_strategy: "all" or "emoji" or "markdown" or "formatting"
        emoji_masks: When token_strategy == "emoji", per-sample (pi_mask, llama_mask)
        markdown_masks: When token_strategy == "markdown", per-sample (pi_mask, llama_mask)
        formatting_masks: When token_strategy == "formatting", per-sample (pi_mask, llama_mask)
    
    Returns:
        steering_vectors: Dict mapping layer -> (hidden_dim,) tensor
    """
    if sample_weights is None:
        sample_weights = [1.0] * len(next(iter(pi_activations.values())))
    
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
    sample_weights_tensor = sample_weights_tensor / sample_weights_tensor.sum()  # Normalize
    
    steering_vectors: Dict[str, torch.Tensor] = {}
    
    for layer_name in pi_activations.keys():
        pi_tensors = pi_activations[layer_name]  # List of (1, seq_len, hidden_dim)
        llama_tensors = llama_activations[layer_name]
        
        # Get hidden dimension from first tensor
        hidden_dim = pi_tensors[0].shape[-1]
        pi_mean = torch.zeros(hidden_dim)
        llama_mean = torch.zeros(hidden_dim)

        if token_strategy == "all":
            # Average across all tokens and samples
            for i, weight in enumerate(sample_weights_tensor):
                # Remove batch dimension and average across sequence dimension
                pi_sample_mean = pi_tensors[i].squeeze(0).mean(dim=0)  # (hidden_dim,)
                llama_sample_mean = llama_tensors[i].squeeze(0).mean(dim=0)  # (hidden_dim,)
                
                pi_mean += weight * pi_sample_mean
                llama_mean += weight * llama_sample_mean
        elif token_strategy == "emoji":
            assert emoji_masks is not None, "emoji_masks must be provided when token_strategy='emoji'"
            included_weight_sum = 0.0
            for i, weight in enumerate(sample_weights_tensor):
                pi_seq = pi_tensors[i].squeeze(0)  # (seq_len, hidden_dim)
                llama_seq = llama_tensors[i].squeeze(0)
                pi_mask, llama_mask = emoji_masks[i]

                # Adjust mask lengths to match activation lengths (activations may include trailing special tokens)
                if pi_mask.numel() != pi_seq.shape[0]:
                    if pi_mask.numel() < pi_seq.shape[0]:
                        pad = torch.zeros(pi_seq.shape[0] - pi_mask.numel(), dtype=torch.bool)
                        pi_mask = torch.cat([pi_mask, pad], dim=0)
                    else:
                        pi_mask = pi_mask[: pi_seq.shape[0]]
                if llama_mask.numel() != llama_seq.shape[0]:
                    if llama_mask.numel() < llama_seq.shape[0]:
                        pad = torch.zeros(llama_seq.shape[0] - llama_mask.numel(), dtype=torch.bool)
                        llama_mask = torch.cat([llama_mask, pad], dim=0)
                    else:
                        llama_mask = llama_mask[: llama_seq.shape[0]]

                # Only include the sample if the Pi side has at least one emoji token
                if not pi_mask.any().item():
                    continue

                pi_sample_mean = pi_seq[pi_mask].mean(dim=0)
                # For Llama: use emoji tokens if present; otherwise fall back to all tokens
                if llama_mask.any().item():
                    llama_sample_mean = llama_seq[llama_mask].mean(dim=0)
                else:
                    llama_sample_mean = llama_seq.mean(dim=0)

                pi_mean += weight * pi_sample_mean
                llama_mean += weight * llama_sample_mean
                included_weight_sum += weight.item()

            if included_weight_sum > 0:
                # Renormalize since we only included a subset of samples
                pi_mean = pi_mean / included_weight_sum
                llama_mean = llama_mean / included_weight_sum
            else:
                print(f"[warn] No samples contained emoji tokens for layer {layer_name}. Vector will be zeros.")
        elif token_strategy == "markdown":
            assert markdown_masks is not None, "markdown_masks must be provided when token_strategy='markdown'"
            included_weight_sum = 0.0
            for i, weight in enumerate(sample_weights_tensor):
                pi_seq = pi_tensors[i].squeeze(0)
                llama_seq = llama_tensors[i].squeeze(0)
                pi_mask, llama_mask = markdown_masks[i]

                # Adjust mask lengths to match activation lengths
                if pi_mask.numel() != pi_seq.shape[0]:
                    if pi_mask.numel() < pi_seq.shape[0]:
                        pad = torch.zeros(pi_seq.shape[0] - pi_mask.numel(), dtype=torch.bool)
                        pi_mask = torch.cat([pi_mask, pad], dim=0)
                    else:
                        pi_mask = pi_mask[: pi_seq.shape[0]]
                if llama_mask.numel() != llama_seq.shape[0]:
                    if llama_mask.numel() < llama_seq.shape[0]:
                        pad = torch.zeros(llama_seq.shape[0] - llama_mask.numel(), dtype=torch.bool)
                        llama_mask = torch.cat([llama_mask, pad], dim=0)
                    else:
                        llama_mask = llama_mask[: llama_seq.shape[0]]

                # Include sample if Llama has any markdown markers
                if not llama_mask.any().item():
                    continue

                # Pi: average over all response tokens
                pi_sample_mean = pi_seq.mean(dim=0)
                # Llama: average only over markdown marker tokens
                llama_sample_mean = llama_seq[llama_mask].mean(dim=0)

                pi_mean += weight * pi_sample_mean
                llama_mean += weight * llama_sample_mean
                included_weight_sum += weight.item()

            if included_weight_sum > 0:
                pi_mean = pi_mean / included_weight_sum
                llama_mean = llama_mean / included_weight_sum
            else:
                print(f"[warn] No samples contained markdown markers for layer {layer_name}. Vector will be zeros.")
        elif token_strategy == "formatting":
            assert formatting_masks is not None, "formatting_masks must be provided when token_strategy='formatting'"
            included_weight_sum = 0.0
            for i, weight in enumerate(sample_weights_tensor):
                pi_seq = pi_tensors[i].squeeze(0)
                llama_seq = llama_tensors[i].squeeze(0)
                pi_mask, llama_mask = formatting_masks[i]

                # Adjust mask lengths to match activation lengths
                if pi_mask.numel() != pi_seq.shape[0]:
                    if pi_mask.numel() < pi_seq.shape[0]:
                        pad = torch.zeros(pi_seq.shape[0] - pi_mask.numel(), dtype=torch.bool)
                        pi_mask = torch.cat([pi_mask, pad], dim=0)
                    else:
                        pi_mask = pi_mask[: pi_seq.shape[0]]
                if llama_mask.numel() != llama_seq.shape[0]:
                    if llama_mask.numel() < llama_seq.shape[0]:
                        pad = torch.zeros(llama_seq.shape[0] - llama_mask.numel(), dtype=torch.bool)
                        llama_mask = torch.cat([llama_mask, pad], dim=0)
                    else:
                        llama_mask = llama_mask[: llama_seq.shape[0]]

                # Include sample if Llama has any formatting characters
                if not llama_mask.any().item():
                    continue

                # Pi: average over all response tokens
                pi_sample_mean = pi_seq.mean(dim=0)
                # Llama: average only over formatting character tokens
                llama_sample_mean = llama_seq[llama_mask].mean(dim=0)

                pi_mean += weight * pi_sample_mean
                llama_mean += weight * llama_sample_mean
                included_weight_sum += weight.item()

            if included_weight_sum > 0:
                pi_mean = pi_mean / included_weight_sum
                llama_mean = llama_mean / included_weight_sum
            else:
                print(f"[warn] No samples contained formatting characters for layer {layer_name}. Vector will be zeros.")
        else:
            raise NotImplementedError(f"Token strategy '{token_strategy}' not implemented")
        
        # Compute steering vector as Pi - Llama
        steering_vectors[layer_name] = pi_mean - llama_mean
    
    return steering_vectors


def save_steering_vectors(steering_vectors: Dict[str, torch.Tensor], output_path: str):
    """Save steering vectors to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(steering_vectors, output_path)
    print(f"Saved steering vectors to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract steering vectors from activations")
    parser.add_argument("--h5-path", required=True, help="Path to h5 activations file")
    parser.add_argument("--output", required=True, help="Output path for steering vectors")
    parser.add_argument("--exclude-samples", nargs="*", type=int, default=[], 
                       help="Sample indices to exclude (test set)")
    parser.add_argument("--token-strategy", default="all", choices=["all", "emoji", "markdown", "formatting"], 
                       help="How to aggregate across tokens")
    parser.add_argument(
        "--dataset-json",
        default="/mnt/vast/home/lawrence/steer-llama/lawrence-generation/traits/elicited/emoji_elicited_filtered.json",
        help="Path to the dataset JSON used to build token masks for emoji/markdown strategies",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="/mnt/vast/share/inf2-training/models/open_source/llama-3.3-70B-Instruct",
        help="Path/name for the tokenizer to match model tokenization",
    )
    
    args = parser.parse_args()

    # Determine training samples (all except excluded)
    discovered = discover_sample_indices(args.h5_path)
    all_samples = set(discovered)
    training_samples = sorted(all_samples - set(args.exclude_samples))
    
    print(f"Training samples: {training_samples}")
    print(f"Test samples: {args.exclude_samples}")
    
    # Load activations
    pi_activations, llama_activations = load_activations(args.h5_path, training_samples)

    emoji_masks: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    markdown_masks: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    formatting_masks: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    tokenizer: Optional[AutoTokenizer] = None

    if args.token_strategy in ("emoji", "markdown", "formatting"):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        pairs = _load_dataset_pairs(args.dataset_json)

    if args.token_strategy == "emoji":
        print("Building emoji token masks using tokenizer...")
        emoji_masks = []
        for idx in training_samples:
            prompt_text, pi_text, llama_text = pairs[idx]
            pi_mask = _build_emoji_mask_for_response(tokenizer, prompt_text, pi_text)
            llama_mask = _build_emoji_mask_for_response(tokenizer, prompt_text, llama_text)
            emoji_masks.append((pi_mask, llama_mask))
        # Quick sanity stats
        pi_counts = [int(m[0].sum().item()) for m in emoji_masks]
        llama_counts = [int(m[1].sum().item()) for m in emoji_masks]
        print(f"Emoji token counts per sample (pi): {pi_counts}")
        print(f"Emoji token counts per sample (llama): {llama_counts}")

    if args.token_strategy == "markdown":
        print("Building markdown token masks using tokenizer...")
        markdown_masks = []
        for idx in training_samples:
            prompt_text, pi_text, llama_text = pairs[idx]
            pi_mask = _build_markdown_mask_for_response(tokenizer, pi_text)
            llama_mask = _build_markdown_mask_for_response(tokenizer, llama_text)
            markdown_masks.append((pi_mask, llama_mask))
        # Quick sanity stats
        pi_counts = [int(m[0].sum().item()) for m in markdown_masks]
        llama_counts = [int(m[1].sum().item()) for m in markdown_masks]
        print(f"Markdown token counts per sample (pi): {pi_counts}")
        print(f"Markdown token counts per sample (llama): {llama_counts}")
        
        # Debug: show actual tokens being averaged for first few samples
        print("\nDebug: Token averaging details for first 3 samples with Llama markdown:")
        for i, idx in enumerate(training_samples[:5]):
            pi_mask, llama_mask = markdown_masks[i]
            if not llama_mask.any().item():
                continue
            prompt_text, pi_text, llama_text = pairs[idx]
            
            # Tokenize to get actual token strings
            pi_tokenized = tokenizer(pi_text, add_special_tokens=False, return_tensors=None)
            llama_tokenized = tokenizer(llama_text, add_special_tokens=False, return_tensors=None)
            
            pi_tokens = [tokenizer.decode([tok_id]) for tok_id in pi_tokenized["input_ids"]]
            llama_tokens = [tokenizer.decode([tok_id]) for tok_id in llama_tokenized["input_ids"]]
            
            print(f"\nSample {idx}:")
            print(f"  Pi text: {repr(pi_text[:100])}{'...' if len(pi_text) > 100 else ''}")
            print(f"  Pi tokens (ALL {len(pi_tokens)} used): {pi_tokens[:10]}{'...' if len(pi_tokens) > 10 else ''}")
            
            print(f"  Llama text: {repr(llama_text[:100])}{'...' if len(llama_text) > 100 else ''}")
            llama_markdown_tokens = [tok for j, tok in enumerate(llama_tokens) if j < len(llama_mask) and llama_mask[j]]
            print(f"  Llama markdown tokens ({len(llama_markdown_tokens)} of {len(llama_tokens)}): {llama_markdown_tokens}")

    if args.token_strategy == "formatting":
        print("Building formatting token masks using tokenizer...")
        formatting_masks = []
        for idx in training_samples:
            prompt_text, pi_text, llama_text = pairs[idx]
            pi_mask = _build_formatting_mask_for_response(tokenizer, pi_text)
            llama_mask = _build_formatting_mask_for_response(tokenizer, llama_text)
            formatting_masks.append((pi_mask, llama_mask))
        # Quick sanity stats
        pi_counts = [int(m[0].sum().item()) for m in formatting_masks]
        llama_counts = [int(m[1].sum().item()) for m in formatting_masks]
        print(f"Formatting token counts per sample (pi): {pi_counts}")
        print(f"Formatting token counts per sample (llama): {llama_counts}")

    # Compute steering vectors
    print("Computing steering vectors...")
    steering_vectors = compute_steering_vectors(
        pi_activations, 
        llama_activations, 
        token_strategy=args.token_strategy,
        emoji_masks=emoji_masks,
        markdown_masks=markdown_masks,
        formatting_masks=formatting_masks,
    )

    # Print some statistics
    print("\nSteering vector statistics:")
    for layer_name, vector in list(steering_vectors.items())[:5]:  # Show first 5 layers
        norm = torch.norm(vector).item()
        mean = vector.mean().item()
        std = vector.std().item()
        print(f"{layer_name}: norm={norm:.4f}, mean={mean:.6f}, std={std:.4f}")
    print("...")
    
    # Save vectors
    save_steering_vectors(steering_vectors, args.output)


if __name__ == "__main__":
    main()