# Steering Vector Experiment Summary

## Successfully Completed

### 1. Data Collection & Processing ✅
- **Source**: 16 prompt-response pairs from Pi and Llama-3.3-70B
- **Activations**: Extracted from `hook_resid_post` at all 80 layers  
- **Storage**: 5.7GB HDF5 file with metadata and activations
- **Test Split**: Samples 0 and 7 reserved for evaluation

### 2. Steering Vector Extraction ✅
```bash
# Successfully extracted steering vectors
uv run python extract_steering_vectors.py \
  --h5-path ../emoji_pi_llama_activations.h5 \
  --output emoji_steering_vectors.pt \
  --exclude-samples 0 7
```

**Results**:
- 80 steering vectors (one per layer)
- Vector norms increase with depth (0.08 → 0.58)
- Format: `Dict[layer_name, torch.Tensor(8192,)]`

### 3. Evaluation Pipeline ✅
- **Coherence Judge**: Qwen3 with binary scoring (0/1)
- **Trait Evaluator**: Qwen3 with 0-100 emoji usage scoring
- **APIs Verified**: Both evaluation endpoints working

## Steering Vector Analysis

### Key Findings
The extracted steering vectors show the expected pattern:

```
Layer 0:  norm=0.0841, mean=-0.000008, std=0.0009
Layer 10: norm=0.4450, mean=0.000025, std=0.0049  
Layer 50: norm=0.5777, mean=-0.000011, std=0.0064
```

- **Increasing Magnitude**: Deeper layers have stronger steering signals
- **Zero-Centered**: Vectors are centered around zero (good for steering)
- **Hidden Dim**: 8192 dimensions matching Llama-3.3-70B architecture

### Expected Steering Effect

**Formula**: `steered_activation = original + α × steering_vector`

**Test Prompts**:
1. "Do you think people text too much these days?"
2. "Is it weird to talk to AI about feelings?"

**Prediction**: Adding Pi-style steering should increase emoji usage from ~5-10 to ~80-90 points.

## Implementation Ready

### Steering Experiment Script
```bash
# Ready to run (needs model access)
srun --gres=gpu:8 uv run python steer_model.py \
  --steering-vectors emoji_steering_vectors.pt \
  --model-path "meta-llama/Llama-3.3-70B-Instruct" \
  --test-prompts "Do you think people text too much these days?" "Is it weird to talk to AI about feelings?" \
  --alphas 0.5 1.0 1.5 2.0 2.5 \
  --output-dir results
```

### Key Features
- **Layer Sweep**: Tests all 80 layers to find optimal intervention point
- **Alpha Sweep**: 5 steering coefficients to balance trait expression vs coherence
- **Evaluation**: Automatic scoring of both coherence and emoji trait expression
- **Visualization**: Generates plots showing steering effectiveness by layer

## Technical Architecture

### Intervention Method
```python
def steering_hook(activations, hook):
    # activations: (batch, seq_len, hidden_dim)
    activations[:, :, :] += alpha * steering_vector
    return activations
```

### Expected Results
- **Best Layer**: Likely layers 40-60 based on norm analysis
- **Optimal Alpha**: Probably 1.0-2.0 for good trait/coherence balance  
- **Trait Improvement**: +70-80 point increase in emoji usage scores

## Next Steps

1. **Model Access**: Need HF token with Llama-3.3-70B access, or use local model path
2. **Full Experiment**: Run complete layer×alpha sweep (400 conditions total)
3. **Analysis**: Identify optimal steering parameters and validate effectiveness
4. **Extension**: Test on broader prompt set and other traits (formatting, playfulness)

The steering vectors are extracted and ready - just need model access to complete the intervention experiment!