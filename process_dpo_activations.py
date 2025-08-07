#!/usr/bin/env python3
"""
Script to extract activations from HookedTransformer for DPO dataset pairs.
Processes single-turn chosen/rejected conversation pairs and stores activations 
at the assistant response divergence point.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

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


def load_dpo_dataset(dataset_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load DPO dataset from JSONL file."""
    chosen_conversations = []
    rejected_conversations = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                conv = json.loads(line)
                chosen_conversations.append(conv["chosen"])
                rejected_conversations.append(conv["rejected"])
    return chosen_conversations, rejected_conversations


def setup_model(config: Dict[str, Any]) -> HookedTransformer:
    """Setup model from configuration."""
    return HookedTransformer.from_pretrained(
        config['model']['name'],
        device=config['model']['device'],
        dtype=getattr(torch, config['model']['dtype']),
    )

def messages_to_tokens(model: HookedTransformer, conversations: List[Dict[str, Any]]) -> List[str]:
    """Convert list of message dictionaries to ChatML text format using model's chat template."""
    text_conversations = []
    for conversation in conversations:
        text_conversations.append(model.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device=model.cfg.device))
    # print(text_conversations[0])
    return text_conversations

def tokenize_conversations(model: HookedTransformer,
                          conversations: List[Dict[str, Any]]) -> Tuple[List[torch.Tensor], List[int]]:
    """Tokenize a conversation. Returns a list of tokens and a list of indices of the assistant response in each conversation."""
    tokens = messages_to_tokens(model, conversations)

    assistant_token_id = model.to_tokens("assistant")[0]
    assistant_response_idxs = [tokens[i].argmax(dim=1) == assistant_token_id for i in range(len(tokens))]
    return tokens, assistant_response_idxs


def get_activations_for_conversation(model: HookedTransformer,
                                   tokens: torch.Tensor,
                                   assistant_response_idx: int) -> Dict[str, np.ndarray]:
    """
    Get activations for a conversation.
    For single-turn DPO, this gets activations for the entire conversation.
    Returns activations for all layers token-by-token.
    """
    
    # Truncate if too long
    # if tokens.shape[1] > max_length:
    #     tokens = tokens[:, :max_length]
    
    # Get activations
    print(tokens.shape)
    print("starting to get activations")
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    
    # Extract activations for all layers
    activations = {}
    for layer_name in cache.keys():
        if "hook_resid_post" in layer_name:
            # Convert to float32 before converting to numpy to avoid BFloat16 issues
            activations[layer_name] = cache[layer_name].float().cpu().numpy()
    
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
            sample_group.attrs['chosen_divergence_idx'] = data['chosen_divergence_idx']
            sample_group.attrs['rejected_divergence_idx'] = data['rejected_divergence_idx']
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
    chosen_conversations, rejected_conversations = load_dpo_dataset(config['dataset']['path'])
    logger.info(f"Loaded {len(chosen_conversations)} chosen conversations and {len(rejected_conversations)} rejected conversations")

    chosen_tokens, assistant_response_idxs = tokenize_conversations(model, chosen_conversations)
    rejected_tokens, assistant_response_idxs = tokenize_conversations(model, rejected_conversations)

    print(chosen_tokens[0].shape)
    print(len(chosen_tokens))
    
    # Process conversations
    activations_data = []

    
    for i in range(len(chosen_tokens)):
        try:
            chosen = chosen_tokens[i]
            rejected = rejected_tokens[i]
            assistant_response_idx = assistant_response_idxs[i]
            
            # Get activations for chosen conversation
            chosen_activations = get_activations_for_conversation(
                model, chosen, assistant_response_idx
            )
            
            # Get activations for rejected conversation
            rejected_activations = get_activations_for_conversation(
                model, rejected, assistant_response_idx
            )
            
            # Store data
            activations_data.append({
                'chosen_length': len(chosen),
                'rejected_length': len(rejected),
                'chosen_activations': chosen_activations,
                'rejected_activations': rejected_activations,
                'response_start_idx': assistant_response_idx,
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
            logger.error(f"Error: {e.with_traceback()}")
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