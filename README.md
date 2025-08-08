# steer-llama (PoC)

Minimal end-to-end pipeline for persona/style vectors (artifact generation → responses → judging → activation extraction → steering), inspired by Persona Vectors [Chen et al., 2025].

Paper: https://arxiv.org/pdf/2507.21509

## Quickstart

1) Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Set API keys (optional; for artifact generation and judging)

```bash
export OPENAI_API_KEY=...
# or
export ANTHROPIC_API_KEY=...
```

3) CLI usage

```bash
python -m steer_llama.cli gen-artifacts --trait "emoji-enthusiastic" \
  --description "Avoid markdown, use emojis, short and upbeat replies"

python -m steer_llama.cli collect \
  --trait "emoji-enthusiastic" \
  --hf-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --pair-index 0

python -m steer_llama.cli judge --trait "emoji-enthusiastic"

python -m steer_llama.cli extract \
  --trait "emoji-enthusiastic" \
  --hf-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --top-k 16 --normalize --select-by-norm

python -m steer_llama.cli steer \
  --trait "emoji-enthusiastic" \
  --hf-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --alpha 2.0 \
  --prompt "Tell me about black holes"
```

Artifacts and outputs are stored under `data/<trait>/`.

## Notes
- Default HF model is small for easy local runs; swap to your target.
- Steering is applied via a forward hook on the selected decoder layer.
- This PoC follows the workflow described in the Persona Vectors paper but keeps each step minimal.