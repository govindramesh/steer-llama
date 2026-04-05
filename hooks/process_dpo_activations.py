#!/usr/bin/env python3
"""
Script to extract activations from HookedTransformer for DPO dataset pairs.
Processes single-turn chosen/rejected conversation pairs and stores activations 
at the assistant response divergence point.

uv run process_dpo_activations.py --config config.yaml
"""

import argparse
from collections import defaultdict
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import gc
import torch
from transformer_lens import HookedTransformer
import yaml
import h5py
import numpy as np
from tqdm import tqdm


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('dpo_activations.log')
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# def load_dpo_dataset(dataset_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
#     """Load DPO dataset from JSONL file."""
#     chosen_conversations = []
#     rejected_conversations = []
#     with open(dataset_path, 'r') as f:
#         for line in f:
#             if line.strip():
#                 conv = json.loads(line)
#                 chosen_conversations.append(conv["chosen"])
#                 rejected_conversations.append(conv["rejected"])
#     return chosen_conversations, rejected_conversations

def load_paired_responses(
    dataset_path: str,
    chosen_field: str = "pi_response",
    rejected_field: str = "llama_response",
    add_system: bool = False,
    system_prompt: str = ""
) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
    """
    Load paired single-turn conversations from a JSON file structured like
    `emoji_elicited_filtered.json`, where the top-level is a mapping:

        prompt: { "llama_response": str, "pi_response": str }

    Returns two aligned lists of conversations, `(chosen_conversations, rejected_conversations)`,
    where each conversation is a list of chat messages in the form expected by
    `tokenizer.apply_chat_template` (dicts with `role` and `content`). The `chosen_field`
    determines which response is treated as chosen, and `rejected_field` as rejected.

    Args:
        dataset_path: Path to the JSON file.
        chosen_field: Key in each record to use as the chosen assistant response.
        rejected_field: Key in each record to use as the rejected assistant response.
        add_system: If True, prepend a system message with `system_prompt`.
        system_prompt: Content for the optional system message.

    Returns:
        Tuple of (chosen_conversations, rejected_conversations).
    """
    logger = logging.getLogger(__name__)
    path_obj = Path(dataset_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(path_obj, "r") as f:
        data = json.load(f)

    # list of lists of dicts with role and content
    chosen_conversations: List[List[Dict[str, str]]] = []
    rejected_conversations: List[List[Dict[str, str]]] = []

    def build_base_messages(prompt_text: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if add_system and system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})
        return messages

    skipped = 0

    for prompt_text, responses in data.items():
        if not isinstance(responses, dict):
            skipped += 1
            continue
        chosen_text = responses.get(chosen_field)
        rejected_text = responses.get(rejected_field)
        if not chosen_text or not rejected_text:
            skipped += 1
            continue

        base = build_base_messages(prompt_text)
        chosen_conversations.append(base + [{"role": "assistant", "content": chosen_text}])
        rejected_conversations.append(base + [{"role": "assistant", "content": rejected_text}])


    if skipped:
        logger.warning(f"Skipped {skipped} records due to missing or invalid fields.")

    logger.info(
        f"Loaded {len(chosen_conversations)} chosen and {len(rejected_conversations)} rejected conversations "
        f"from {dataset_path} using chosen='{chosen_field}', rejected='{rejected_field}'."
    )

    return chosen_conversations, rejected_conversations


def setup_model(config: Dict[str, Any]) -> HookedTransformer:
    """Setup model from configuration."""
    model_config = config['model']
    return HookedTransformer.from_pretrained(
        model_config['name'],
        device=model_config['device'],
        n_devices=model_config['n_devices'],
        dtype=getattr(torch, model_config['dtype']),
        name_or_path=model_config['path'],
        move_to_device=True,
    )


def messages_to_tokens(model: HookedTransformer, conversations: List[List[Dict[str, Any]]]) -> List[torch.Tensor]:
    """Convert list of message dictionaries to ChatML text format using model's chat template."""
    text_conversations = []
    # For multi-GPU setups, use the device of the first layer (embed layer)
    for conversation in conversations:
        tokens = model.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        text_conversations.append(tokens)
    return text_conversations


def tokenize_conversations(model: HookedTransformer,
                          conversations: List[List[Dict[str, Any]]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Tokenize a conversation. Returns a list of tokens and a list of indices of the assistant response in each conversation."""
    tokens = messages_to_tokens(model, conversations)

    tokenizer = model.tokenizer

    # Build the pattern corresponding to: <|start_header_id|>assistant<|end_header_id|>
    pattern = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)

    # Newline after header is typically a single token; detect it robustly
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)

    assistant_response_idxs: List[torch.Tensor] = []
    for i in range(len(tokens)):
        token_sequence = tokens[i][0]  # Remove batch dimension
        seq_list = token_sequence.tolist()
        m = len(pattern)

        # Find all occurrences of the assistant header pattern
        match_indices = []
        for j in range(0, len(seq_list) - m + 1):
            if seq_list[j:j + m] == pattern:
                match_indices.append(j)
        if not match_indices:
            raise ValueError("Assistant header pattern '<|start_header_id|>assistant<|end_header_id|>' not found in tokenized conversation.")

        # Use the penultimate occurrence to avoid the trailing generation prompt header
        header_idx = match_indices[-2] if len(match_indices) >= 2 else match_indices[-1]

        start_idx = header_idx + m
        # Skip a single newline token immediately after header if present
        if newline_ids and len(newline_ids) == 1 and start_idx < len(seq_list) and seq_list[start_idx] == newline_ids[0]:
            start_idx += 1

        assistant_response_idxs.append(torch.tensor(start_idx, device=token_sequence.device))

    return tokens, assistant_response_idxs


def get_activations_for_conversation(model: HookedTransformer,
                                   tokens: torch.Tensor,
                                   assistant_response_idx: int) -> Dict[str, np.ndarray]:
    """
    Get activations for a conversation via a single forward pass.
    Caches only positions from the assistant response start onward and returns
    activations per layer as CPU float32 numpy arrays.
    """

    # Ensure tokens are on the model's expected device for forward
    tokens = tokens.to(model.cfg.device)

    activations: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        # Cache activations from all layers
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name.endswith("hook_resid_post"),
            return_type=None,
        )

        for layer_name in cache.keys():
            # Slice to only include positions from assistant response onward
            layer_activations = cache[layer_name][:, assistant_response_idx:, :]
            # Move to CPU and convert to float32 for HDF5 compatibility
            tensor_cpu = layer_activations.cpu().to(dtype=torch.float32)
            activations[layer_name] = tensor_cpu.numpy()

        # Let Python GC handle cache dict lifecycle
        del cache

    return activations


def save_activations_to_h5(activations_data: List[Dict[str, Any]], 
                          output_file: str,
                          compression: str = "gzip",
                          compression_opts: int = 9) -> None:
    """Save activations data to HDF5 file."""
    with h5py.File(output_file, 'w') as f:
        # Create groups for metadata and activations
        metadata_group = f.create_group('metadata')
        activations_group = f.create_group('activations')
        
        # Save metadata
        for i, data in enumerate(activations_data):
            sample_group = metadata_group.create_group(f'sample_{i}')
            sample_group.attrs['response_start_idx'] = data['response_start_idx'].cpu().item()
            sample_group.attrs['chosen_length'] = data['chosen_length']
            sample_group.attrs['rejected_length'] = data['rejected_length']
            
            # Save activations
            chosen_group = activations_group.create_group(f'sample_{i}_chosen')
            rejected_group = activations_group.create_group(f'sample_{i}_rejected')
            
            for layer_name, activations in data['chosen_activations'].items():
                chosen_group.create_dataset(
                    layer_name, 
                    data=activations,
                    compression=compression,
                    compression_opts=compression_opts
                )
            
            for layer_name, activations in data['rejected_activations'].items():
                rejected_group.create_dataset(
                    layer_name, 
                    data=activations,
                    compression=compression,
                    compression_opts=compression_opts
                )


def main():
    parser = argparse.ArgumentParser(description="Extract activations from DPO dataset")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Initialize model
    logger.info(f"Loading model: {config['model']['name']}")
    model = setup_model(config)
    
    # Load dataset
    logger.info(f"Loading dataset from {config['dataset']['path']}")
    dataset_cfg = config.get('dataset', {})
    chosen_conversations, rejected_conversations = load_paired_responses(
        dataset_cfg['path'],
        chosen_field=dataset_cfg.get('chosen_field', 'pi_response'),
        rejected_field=dataset_cfg.get('rejected_field', 'llama_response'),
        add_system=dataset_cfg.get('add_system', False),
        system_prompt=dataset_cfg.get('system_prompt', '')
    )
    logger.info(f"Loaded {len(chosen_conversations)} chosen conversations and {len(rejected_conversations)} rejected conversations")

    chosen_tokens, chosen_assistant_response_idxs = tokenize_conversations(model, chosen_conversations)
    rejected_tokens, rejected_assistant_response_idxs = tokenize_conversations(model, rejected_conversations)

    # print(chosen_tokens[0].shape)
    # print(len(chosen_tokens))
    
    # Process conversations
    activations_data = []

    
    for i in tqdm(range(len(chosen_tokens))):
        try:
            chosen = chosen_tokens[i]
            rejected = rejected_tokens[i]
            chosen_assistant_response_idx = chosen_assistant_response_idxs[i]
            rejected_assistant_response_idx = rejected_assistant_response_idxs[i]
            # print("chosen_assistant_response_idx", chosen_assistant_response_idx)
            # print("rejected_assistant_response_idx", rejected_assistant_response_idx)
            
            print(f"="*100)
            print(f"Processing conversation {i}")
            print(f"="*100)
            # Decode the tensors (assumed shape (1, seq_len)) to text using the model's tokenizer
            chosen_decoded = model.tokenizer.decode(chosen[0].tolist())
            rejected_decoded = model.tokenizer.decode(rejected[0].tolist())
            print(f"chosen (decoded): {chosen_decoded}")
            print(f"rejected (decoded): {rejected_decoded}")
            print(f"chosen_assistant_response_idx: {chosen_assistant_response_idx}")
            print(f"rejected_assistant_response_idx: {rejected_assistant_response_idx}")
            print(f"="*100)
            # Get activations for chosen conversation
            chosen_activations = get_activations_for_conversation(
                model, chosen, chosen_assistant_response_idx
            )
            
            # Get activations for rejected conversation
            rejected_activations = get_activations_for_conversation(
                model, rejected, rejected_assistant_response_idx
            )
            
            # Store data
            activations_data.append({
                'chosen_length': chosen.shape[1],
                'rejected_length': rejected.shape[1],
                'chosen_activations': chosen_activations,
                'rejected_activations': rejected_activations,
                'response_start_idx': chosen_assistant_response_idx,
            })
            
            # Save intermediate results if enabled
            if config['processing']['save_intermediate'] and (i + 1) % 10 == 0:
                intermediate_file = f"activations_intermediate_{i+1}.h5"
                save_activations_to_h5(
                    activations_data, 
                    intermediate_file,
                    config['output']['compression'],
                    config['output']['compression_opts']
                )
                logger.info(f"Saved intermediate results to {intermediate_file}")
                
        except Exception as e:
            logger.error(f"Error processing conversation {i}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue
    
    # Save final results
    logger.info(f"Saving final results to {config['output']['file']}")
    save_activations_to_h5(
        activations_data,
        config['output']['file'],
        config['output']['compression'],
        config['output']['compression_opts']
    )
    
    logger.info(f"Processing complete. Processed {len(activations_data)} conversations.")


if __name__ == "__main__":
    main() 