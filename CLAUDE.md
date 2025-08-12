# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project implementing a "steer-llama" pipeline for style transfer in language models using persona/steering vectors. The project extracts activations from transformer models to create vectors that can transfer specific traits (like emoji usage, formatting preferences, playful tone) from one model (Pi) to another (Llama-3.3-70B).

## Architecture

### Core Components

- **Activation Extraction Pipeline**: `govind-hooks/process_dpo_activations.py` - Main script that extracts activations from HookedTransformer models using DPO dataset pairs
- **Trait Generation**: `lawrence-generation/filter.py` - Generates trait elicitation prompts, questions, and evaluation criteria using structured output models
- **Response Generation & Evaluation**: `lawrence-generation/generate.py` - Compares Pi and Llama responses, judges them using evaluation prompts, and filters based on score differences
- **Data Processing**: `lawrence-generation/merge.py` - Utility to merge filtered results

### Data Flow

1. **Trait Definition**: Define traits in JSON files (`lawrence-generation/traits/`) with trait instructions
2. **Prompt Generation**: Use `filter.py` to generate elicitation prompts and evaluation questions
3. **Response Collection**: Use `generate.py` to collect Pi and Llama responses, judge them, and filter by score difference (threshold: 40)
4. **Activation Extraction**: Use `process_dpo_activations.py` to extract transformer activations at divergence points
5. **Vector Creation**: Process activations to create steering vectors (implementation not yet complete)

### Key Dependencies

- **transformer-lens**: For HookedTransformer model loading and activation extraction
- **taster**: Custom library for Pi API integration and VLLM models (local dependency)
- **torch**: PyTorch for model operations
- **h5py**: HDF5 format for storing large activation matrices
- **pyyaml**: Configuration file parsing

## Development Commands

### Environment Setup
```bash
uv sync
```

### GPU allocation

Use salloc and use 8 GPUs for Llama-3.3-70B.

### Running the Pipeline

1. **Generate trait elicitation prompts**:
```bash
uv run python lawrence-generation/filter.py
```

2. **Collect and evaluate responses**:
```bash
uv run python lawrence-generation/generate.py
```

3. **Extract activations**:
```bash
uv run python govind-hooks/process_dpo_activations.py --config govind-hooks/config.yaml
```

### Configuration

- Main config: `govind-hooks/config.yaml` - Controls model settings, dataset paths, and processing options
- Supports multi-GPU processing (configured for 8 GPUs with `srun --gres=gpu:8 --pty bash`)
- Model path: `/mnt/vast/share/inf2-training/models/open_source/llama-3.3-70B-Instruct`

### Data Structures

- **Trait definitions**: JSON files in `lawrence-generation/traits/` with `name` and `trait_instruction` fields
- **Filtered results**: Format `{prompt: {"pi_response": str, "llama_response": str}}`
- **Activations**: Stored in HDF5 with metadata (divergence indices, lengths) and layer-wise activations

### API Endpoints

The project integrates with several model endpoints:
- Pi API: `https://api.inflection.ai/external/api/inference/openai/v1/chat/completions`
- Qwen (local): `http://172.28.127.109:8010/v1/chat/completions`
- Llama (local): `http://172.28.127.202:8000/v1/chat/completions`

### Development Environment

- Always use `uv run` for Python script execution
- Use `srun --gres=gpu:8 --pty bash` for GPU allocation on SLURM systems
- Intermediate saves enabled every 10 samples during activation extraction
- Comprehensive logging to `dpo_activations.log`