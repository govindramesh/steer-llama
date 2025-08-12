#!/usr/bin/env python3
"""
Extract steering vectors from Pi vs Llama activations.
Computes mean difference vectors across samples and tokens for each layer.
"""

import argparse
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


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


def compute_steering_vectors(
    pi_activations: Dict[str, List[torch.Tensor]],
    llama_activations: Dict[str, List[torch.Tensor]],
    sample_weights: Optional[List[float]] = None,
    token_strategy: str = "all"
) -> Dict[str, torch.Tensor]:
    """
    Compute steering vectors as Pi_mean - Llama_mean for each layer.
    
    Args:
        pi_activations: Dict mapping layer -> List[tensor] (each tensor is (1, seq_len, hidden_dim))
        llama_activations: Same format
        sample_weights: Optional weights for each sample (defaults to uniform)
        token_strategy: "all" or "response_start" (for now, implement "all")
    
    Returns:
        steering_vectors: Dict mapping layer -> (hidden_dim,) tensor
    """
    if sample_weights is None:
        sample_weights = [1.0] * len(next(iter(pi_activations.values())))
    
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    sample_weights = sample_weights / sample_weights.sum()  # Normalize
    
    steering_vectors = {}
    
    for layer_name in pi_activations.keys():
        pi_tensors = pi_activations[layer_name]  # List of (1, seq_len, hidden_dim)
        llama_tensors = llama_activations[layer_name]
        
        if token_strategy == "all":
            # Average across all tokens and samples
            # Get hidden dimension from first tensor
            hidden_dim = pi_tensors[0].shape[-1]
            pi_mean = torch.zeros(hidden_dim)
            llama_mean = torch.zeros(hidden_dim)
            
            for i, weight in enumerate(sample_weights):
                # Average across tokens for this sample, then weight by sample weight
                # Remove batch dimension and average across sequence dimension
                pi_sample_mean = pi_tensors[i].squeeze(0).mean(dim=0)  # (hidden_dim,)
                llama_sample_mean = llama_tensors[i].squeeze(0).mean(dim=0)  # (hidden_dim,)
                
                pi_mean += weight * pi_sample_mean
                llama_mean += weight * llama_sample_mean
        
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
    parser.add_argument("--exclude-samples", nargs="*", type=int, default=[0, 7], 
                       help="Sample indices to exclude (test set)")
    parser.add_argument("--token-strategy", default="all", choices=["all"], 
                       help="How to aggregate across tokens")
    
    args = parser.parse_args()
    
    # Determine training samples (all except excluded)
    with h5py.File(args.h5_path, 'r') as f:
        all_samples = set(range(16))  # We know there are 16 samples
        training_samples = sorted(all_samples - set(args.exclude_samples))
    
    print(f"Training samples: {training_samples}")
    print(f"Test samples: {args.exclude_samples}")
    
    # Load activations
    pi_activations, llama_activations = load_activations(args.h5_path, training_samples)
    
    # Compute steering vectors
    print("Computing steering vectors...")
    steering_vectors = compute_steering_vectors(
        pi_activations, 
        llama_activations, 
        token_strategy=args.token_strategy
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