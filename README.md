# DPO Activations Extractor

This script extracts activations from a HookedTransformer model for DPO (Direct Preference Optimization) dataset pairs. It processes chosen/rejected conversation pairs and stores activations at the point where conversations diverge.

## Features

- Loads DPO dataset in ChatML format
- Finds divergence points between chosen and rejected conversations
- Extracts token-by-token activations from all transformer layers
- Saves activations to HDF5 format with compression
- Supports intermediate saving for long-running processes
- Comprehensive logging and error handling

## Installation

1. Install dependencies:
```bash
uv sync
```

2. Update the configuration file `config.yaml` with your dataset path and model settings.

## Usage

### Basic Usage
```bash
python process_dpo_activations.py
```

### With Custom Config
```bash
python process_dpo_activations.py --config my_config.yaml
```

### With Different Log Level
```bash
python process_dpo_activations.py --log-level DEBUG
```

## Configuration

Edit `config.yaml` to customize:

- **Model settings**: Model name, device, dtype
- **Dataset path**: Path to your DPO dataset JSONL file
- **Output settings**: Output file path, compression options
- **Processing settings**: Batch size, max sequence length

### Example Dataset Format

Your DPO dataset should be in JSONL format with each line containing a single-turn conversation pair:
```json
{
  "chosen": [
    {"role": "user", "content": "What is the capital of France?"}, 
    {"role": "assistant", "content": "Paris is the capital of France."}
  ], 
  "rejected": [
    {"role": "user", "content": "What is the capital of France?"}, 
    {"role": "assistant", "content": "I don't know."}
  ]
}
```

**Note**: Each sample should contain exactly one user message and one assistant response. The user messages should be identical between chosen and rejected pairs, with only the assistant responses differing.

## Output Format

The script creates an HDF5 file with the following structure:

```
activations_output.h5/
├── metadata/
│   ├── sample_0/
│   │   ├── chosen_divergence_idx
│   │   ├── rejected_divergence_idx
│   │   ├── chosen_length
│   │   └── rejected_length
│   └── ...
└── activations/
    ├── sample_0_chosen/
    │   ├── blocks.0.hook_resid_post
    │   ├── blocks.1.hook_resid_post
    │   └── ...
    ├── sample_0_rejected/
    │   ├── blocks.0.hook_resid_post
    │   ├── blocks.1.hook_resid_post
    │   └── ...
    └── ...
```

## Loading Activations

```python
import h5py

with h5py.File('activations_output.h5', 'r') as f:
    # Get metadata for sample 0
    metadata = f['metadata/sample_0']
    divergence_idx = metadata.attrs['chosen_divergence_idx']
    
    # Get activations for chosen conversation
    chosen_activations = f['activations/sample_0_chosen']
    layer_0_activations = chosen_activations['blocks.0.hook_resid_post'][:]
```

## Logging

The script creates a log file `dpo_activations.log` with detailed information about the processing progress and any errors encountered.

## Error Handling

- Invalid conversations are skipped with error logging
- Memory issues are handled gracefully
- Intermediate saves prevent data loss on long runs