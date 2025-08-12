#!/usr/bin/env python3
"""
Script to inspect HDF5 files containing activations data.
Shows structure, number of items, and tensor shapes.
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import List


def _print_sample_group(sample_group: h5py.Group, sample_label: str, max_layers: int) -> None:
    print(f"Detailed view of {sample_label}:")
    print(f"  Contains {len(sample_group)} layer activations:")

    total_size_mb = 0.0
    printed = 0
    for layer_name in sorted(sample_group.keys()):
        dataset = sample_group[layer_name]
        shape = dataset.shape
        dtype = dataset.dtype
        size_mb = dataset.size * dataset.dtype.itemsize / (1024**2)
        total_size_mb += size_mb

        if printed < max_layers:
            print(f"    {layer_name}:")
            print(f"      Shape: {shape}")
            print(f"      Dtype: {dtype}")
            print(f"      Size: {size_mb:.2f} MB")
            print(f"      Elements: {dataset.size:,}")
            printed += 1
    if len(sample_group.keys()) > max_layers:
        print(f"    ... {len(sample_group.keys()) - max_layers} more layers")

    print(f"  Total size for this sample: {total_size_mb:.2f} MB")
    print()


def inspect_h5_file(file_path: str, list_all: bool, sample_idx: int, max_layers: int):
    """Inspect HDF5 file and print structure and shapes."""
    if not Path(file_path).exists():
        print(f"Error: File {file_path} does not exist")
        return

    print(f"Inspecting HDF5 file: {file_path}")
    print("=" * 60)

    # Try to load the corresponding dataset file to show prompts
    dataset_file = "/mnt/vast/home/lawrence/steer-llama/lawrence-generation/traits/elicited/emoji_elicited_filtered.json"
    prompts = []
    try:
        import json
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        prompts = list(data.keys())
        print(f"Found corresponding dataset file with {len(prompts)} prompts")
        print("Prompts used:")
        for i, prompt in enumerate(prompts):
            print(f"  {i}: {prompt}")
        print()
    except Exception as e:
        print(f"Could not load dataset file: {e}")
        print()

    with h5py.File(file_path, 'r') as f:
        print(f"File size: {Path(file_path).stat().st_size / (1024**2):.2f} MB")
        print(f"Root groups: {list(f.keys())}")
        print()

        # Inspect metadata
        num_meta = 0
        if 'metadata' in f:
            metadata_group = f['metadata']
            num_meta = len(metadata_group)
            print(f"Metadata group contains {num_meta} samples:")

            for sample_name in sorted(metadata_group.keys()):
                sample = metadata_group[sample_name]
                print(f"  {sample_name}:")
                for attr_name in sample.attrs:
                    print(f"    {attr_name}: {sample.attrs[attr_name]}")
            print()

        # Inspect activations
        if 'activations' in f:
            activations_group = f['activations']
            all_sample_groups: List[str] = sorted(activations_group.keys())
            print(f"Activations group contains {len(activations_group)} sample groups:")

            # Count chosen and rejected samples
            chosen_samples = [k for k in all_sample_groups if 'chosen' in k]
            rejected_samples = [k for k in all_sample_groups if 'rejected' in k]

            print(f"  - {len(chosen_samples)} chosen samples")
            print(f"  - {len(rejected_samples)} rejected samples")
            print()

            # Per-sample listing
            if list_all:
                print("Per-sample layer counts (first few layers shown):")
                for sample_name in all_sample_groups:
                    group = activations_group[sample_name]
                    print(f"  {sample_name}: {len(group)} layers")
                    _print_sample_group(group, sample_name, max_layers)

            # Inspect a specific sample index across chosen/rejected
            if sample_idx >= 0:
                chosen_key = f"sample_{sample_idx}_chosen"
                rejected_key = f"sample_{sample_idx}_rejected"
                if chosen_key in activations_group:
                    _print_sample_group(activations_group[chosen_key], chosen_key, max_layers)
                else:
                    print(f"No group found: {chosen_key}")
                if rejected_key in activations_group:
                    _print_sample_group(activations_group[rejected_key], rejected_key, max_layers)
                else:
                    print(f"No group found: {rejected_key}")

            # Show summary statistics from first chosen sample
            if chosen_samples:
                print("Summary across all samples:")

                first_sample = activations_group[sorted(chosen_samples)[0]]
                layer_names = sorted(first_sample.keys())

                print(f"  Layers found in first sample: {len(layer_names)}")
                print(f"  Layer names (first 5): {layer_names[:5]}{'...' if len(layer_names) > 5 else ''}")

                # Check shapes consistency for a few layers across a few samples
                print("\n  Shape consistency quick check:")
                for layer_name in layer_names[:3]:  # Check first 3 layers
                    shapes = []
                    for sample_name in sorted(chosen_samples)[:5]:  # Check first 5 samples
                        if sample_name in activations_group and layer_name in activations_group[sample_name]:
                            shapes.append(activations_group[sample_name][layer_name].shape)
                    if shapes:
                        consistent = all(s == shapes[0] for s in shapes)
                        print(f"    {layer_name}: {shapes[0]} {'✓' if consistent else '✗ (inconsistent)'}")


def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 activations file")
    parser.add_argument("file_path", type=str, help="Path to HDF5 file to inspect")
    parser.add_argument("--all", dest="list_all", action="store_true", help="List layer shapes for all samples")
    parser.add_argument("--sample", type=int, default=-1, help="Inspect a specific sample index in detail")
    parser.add_argument("--max-layers", type=int, default=5, help="Max layers to print per sample")

    args = parser.parse_args()
    inspect_h5_file(args.file_path, args.list_all, args.sample, args.max_layers)


if __name__ == "__main__":
    main() 