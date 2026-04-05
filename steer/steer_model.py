#!/usr/bin/env python3
"""
Steering experiment script using TransformerLens.
Tests steering effectiveness across layers and alpha values.
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from pydantic import BaseModel
from transformer_lens import HookedTransformer


@dataclass
class SteeringResult:
    layer: int
    alpha: float
    prompt: str
    response: str
    coherence_score: int
    trait_score: float
    

class CoherenceJudgment(BaseModel):
    justification: str
    answer: int  # 0 or 1


def load_steering_vectors(vectors_path: str) -> Dict[str, torch.Tensor]:
    """Load steering vectors from file."""
    return torch.load(vectors_path, map_location='cpu')


def create_steering_hook(steering_vector: torch.Tensor, alpha: float):
    """Create a hook function that adds steering vector to residual stream."""
    def steering_hook(activations, hook):
        # activations shape: (batch_size, seq_len, hidden_dim)
        # Ensure steering vector matches activation device and dtype; operate out-of-place
        vector = steering_vector.to(device=activations.device, dtype=activations.dtype)
        return activations + alpha * vector
    return steering_hook

def hooked_generate(hooked_model: HookedTransformer, input_ids, max_tokens, layer_name, hook_fn, temperature=0.7, do_sample=True, top_p=0.9, **kwargs):
    """Custom generation loop using hooked forward passes."""
    # Convert input_ids to the model device if not already
    input_ids = input_ids.to(hooked_model.cfg.device)
    
    # Start with the initial input tokens
    generated_ids = input_ids.clone()
    
    for _ in range(max_tokens):
        # Run forward pass with hooks
        with hooked_model.hooks(fwd_hooks=[(layer_name, hook_fn)]):
            with torch.no_grad():
                logits = hooked_model(generated_ids)
        
        # Apply temperature scaling
        if do_sample and temperature != 1.0:
            logits = logits / temperature
        
        # Get next token
        if do_sample:
            # Apply top-p (nucleus) sampling if specified
            if top_p < 1.0:
                # Sort logits and get cumulative probabilities
                sorted_logits, sorted_indices = torch.sort(logits[0, -1], descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[0] = False
                
                # Set logits of removed tokens to -inf
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, -1, indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = torch.softmax(logits[0, -1], dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            # Greedy sampling
            next_token = logits[0, -1].argmax().unsqueeze(0)
        
        # Reshape to match expected dimensions
        next_token = next_token.unsqueeze(0)
        
        # Append the new token
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # Check for EOS token (optional - depends on your tokenizer)
        if hasattr(hooked_model.tokenizer, 'eos_token_id') and next_token.item() == hooked_model.tokenizer.eos_token_id:
            break
    
    return generated_ids


def generate_steered_response(
    model: HookedTransformer,
    tokenizer,
    prompt: str,
    layer_name: str,
    steering_vector: torch.Tensor,
    alpha: float,
    max_tokens: int = 150,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: float = 0.9
) -> str:
    """Generate response with steering applied at specified layer."""
    
    # Prepare input
    try:
        # Try to use chat template if available
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (ValueError, AttributeError):
        # Fallback to raw prompt if no chat template
        print("Warning: No chat template available, using raw prompt")
        input_text = prompt
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.cfg.device)
    
    # Add steering hook
    hook_fn = create_steering_hook(steering_vector, alpha)
    
    # Generate with steering using custom loop
    with torch.no_grad():
        generated_ids = hooked_generate(
            model, 
            input_ids, 
            max_tokens, 
            layer_name, 
            hook_fn,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p
        )
    
    # Decode response (skip original input)
    response = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate_coherence(response: str, prompt: str) -> int:
    """Evaluate response coherence using Qwen3 API."""
    
    coherence_prompt = f"""You are evaluating whether a response is coherent and makes sense.

Prompt: {prompt}
Response: {response}

A coherent response should:
1. Be grammatically correct and well-structured
2. Make logical sense and stay on topic
3. Be appropriate for the given prompt
4. Not contain contradictions or nonsensical statements

Respond with a JSON object containing:
- justification: Brief explanation of your assessment
- answer: 0 if incoherent, 1 if coherent"""

    try:
        response_obj = requests.post(
            "http://172.28.127.109:8010/v1/chat/completions",
            json={
                "model": "qwen3-235b-reasoning",
                "messages": [{"role": "user", "content": coherence_prompt}],
                "temperature": 0.0,
                "max_tokens": 200
            },
            timeout=30
        )
        
        if response_obj.status_code == 200:
            result = response_obj.json()
            content = result['choices'][0]['message']['content']
            
            # Parse JSON response
            judgment = CoherenceJudgment.model_validate_json(content)
            return judgment.answer
        else:
            print(f"Coherence API error: {response_obj.status_code}")
            return 1  # Default to coherent if API fails
            
    except Exception as e:
        print(f"Coherence evaluation error: {e}")
        return 1  # Default to coherent if evaluation fails


def evaluate_trait_expression(response: str, prompt: str) -> float:
    """Evaluate emoji trait expression using the existing eval prompt."""
    
    eval_prompt_template = """You are evaluating whether the model's response displays the trait: emoji usage.
The model tends to use emojis in its responses, particularly when the user uses emojis or expresses emotion, excitement, or casual tone. The use of emojis should feel natural and contextually appropriate, enhancing the expressiveness of the response.
Prompt:
[QUESTION START]
{question}
[QUESTION END]
Model response:
[ANSWER START]
{answer}
[ANSWER END]
Focus only on whether the model exhibits this trait.
Respond with:
- REFUSAL if the model clearly refused to answer
- A score from 0 (trait not present at all) to 100 (trait strongly present)
Respond with a number between 0 and 100. Don't say anything else, just the number."""
    
    eval_prompt = eval_prompt_template.format(question=prompt, answer=response)
    
    try:
        response_obj = requests.post(
            "http://172.28.127.109:8010/v1/chat/completions",
            json={
                "model": "qwen3-235b-reasoning",
                "messages": [{"role": "user", "content": eval_prompt}],
                "temperature": 0.0,
                "max_tokens": 10
            },
            timeout=30
        )
        
        if response_obj.status_code == 200:
            result = response_obj.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Extract numeric score
            if content == "REFUSAL":
                return 0.0
            else:
                return float(content)
        else:
            print(f"Trait API error: {response_obj.status_code}")
            return 0.0
            
    except Exception as e:
        print(f"Trait evaluation error: {e}")
        return 0.0


def run_steering_experiment(
    model: HookedTransformer,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    test_prompts: List[str],
    alphas: List[float],
    output_dir: str
) -> List[SteeringResult]:
    """Run steering experiment across all layers and alpha values."""
    
    results = []
    
    # Get layer names (sorted)
    layer_names = sorted(steering_vectors.keys())
    
    print(f"Testing {len(layer_names)} layers × {len(alphas)} alphas × {len(test_prompts)} prompts = {len(layer_names) * len(alphas) * len(test_prompts)} total conditions")
    
    for layer_name in layer_names:
        layer_num = int(layer_name.split('.')[1])  # Extract number from "blocks.X.hook_resid_post"
        steering_vector = steering_vectors[layer_name]
        
        print(f"\nTesting layer {layer_num} ({layer_name})...")
        
        for alpha in alphas:
            print(f"  Alpha = {alpha}")
            
            for prompt in test_prompts:
                print(f"    Prompt: {prompt[:50]}...")
                
                # Generate steered response
                response = generate_steered_response(
                    model, tokenizer, prompt, layer_name, steering_vector, alpha
                )
                
                # Evaluate response
                coherence = evaluate_coherence(response, prompt)
                trait_score = evaluate_trait_expression(response, prompt) if coherence == 1 else 0.0
                
                result = SteeringResult(
                    layer=layer_num,
                    alpha=alpha,
                    prompt=prompt,
                    response=response,
                    coherence_score=coherence,
                    trait_score=trait_score
                )
                
                results.append(result)
                
                print(f"      Coherence: {coherence}, Trait: {trait_score:.1f}")
    
    return results


def plot_results(results: List[SteeringResult], output_path: str):
    """Create visualization of steering results."""
    
    # Filter to coherent responses only
    coherent_results = [r for r in results if r.coherence_score == 1]
    
    if not coherent_results:
        print("No coherent results to plot!")
        return
    
    # Group by layer and alpha, average trait scores
    layer_alpha_scores = {}
    for result in coherent_results:
        key = (result.layer, result.alpha)
        if key not in layer_alpha_scores:
            layer_alpha_scores[key] = []
        layer_alpha_scores[key].append(result.trait_score)
    
    # Average scores
    plot_data = {}
    for (layer, alpha), scores in layer_alpha_scores.items():
        plot_data[(layer, alpha)] = np.mean(scores)
    
    # Get unique layers and alphas
    layers = sorted(set(layer for layer, alpha in plot_data.keys()))
    alphas = sorted(set(alpha for layer, alpha in plot_data.keys()))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    for alpha in alphas:
        layer_scores = []
        for layer in layers:
            score = plot_data.get((layer, alpha), 0.0)
            layer_scores.append(score)
        
        plt.plot(layers, layer_scores, marker='o', label=f'α = {alpha}')
    
    plt.xlabel('Layer')
    plt.ylabel('Trait Expression Score (0-100)')
    plt.title('Steering Effectiveness Across Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def save_results(results: List[SteeringResult], output_path: str):
    """Save detailed results to JSON."""
    results_data = []
    for r in results:
        results_data.append({
            "layer": r.layer,
            "alpha": r.alpha,
            "prompt": r.prompt,
            "response": r.response,
            "coherence_score": r.coherence_score,
            "trait_score": r.trait_score
        })
    
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Saved detailed results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run steering experiment")
    parser.add_argument("--steering-vectors", required=True, help="Path to steering vectors file")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--test-prompts", nargs="+", required=True, help="Test prompts")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    import os
    hf_token = os.getenv("HF_KEY")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print(f"Using HF token for model access...")
    
    model = HookedTransformer.from_pretrained_no_processing(
        "meta-llama/Meta-Llama-3-70B-Instruct",
        device="cuda",
        n_devices=8,
        dtype=torch.bfloat16,
        name_or_path=args.model_path,
        move_to_device=True
    )
    tokenizer = model.tokenizer
    
    # Load steering vectors
    print("Loading steering vectors...")
    steering_vectors = load_steering_vectors(args.steering_vectors)
    
    # Run experiment
    print("Running steering experiment...")
    results = run_steering_experiment(
        model, tokenizer, steering_vectors, args.test_prompts, args.alphas, args.output_dir
    )
    
    # Save results
    results_path = Path(args.output_dir) / "results.json"
    save_results(results, str(results_path))
    
    # Create plot
    plot_path = Path(args.output_dir) / "steering_results.png"
    plot_results(results, str(plot_path))
    
    # Print summary
    coherent_count = sum(1 for r in results if r.coherence_score == 1)
    total_count = len(results)
    print(f"\nSummary: {coherent_count}/{total_count} responses were coherent")
    
    if coherent_count > 0:
        coherent_results = [r for r in results if r.coherence_score == 1]
        avg_trait_score = np.mean([r.trait_score for r in coherent_results])
        print(f"Average trait score (coherent responses): {avg_trait_score:.1f}")


if __name__ == "__main__":
    main()