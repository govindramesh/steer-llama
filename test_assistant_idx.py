#!/usr/bin/env python3
"""
Test script to verify assistant response index calculation without loading the full model.
"""

import torch
from transformers import AutoTokenizer
import json

def test_assistant_response_index():
    """Test the assistant response index calculation logic."""
    
    # Load just the tokenizer (much lighter than the full model)
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/vast/share/inf2-training/models/open_source/llama-3.3-70B-Instruct"
    )
    
    # Load a sample conversation
    with open("/mnt/vast/home/lawrence/steer-llama/lawrence-generation/traits/elicited/emoji_elicited_filtered.json", 'r') as f:
        data = json.load(f)
    
    # Take the first sample
    first_prompt = list(data.keys())[0]
    first_responses = data[first_prompt]
    
    # Create conversation format
    chosen_conversation = [
        {"role": "user", "content": first_prompt},
        {"role": "assistant", "content": first_responses["pi_response"]}
    ]
    
    rejected_conversation = [
        {"role": "user", "content": first_prompt},
        {"role": "assistant", "content": first_responses["llama_response"]}
    ]
    
    print("=== CHOSEN CONVERSATION ===")
    print(f"User: {first_prompt}")
    print(f"Assistant: {first_responses['pi_response']}")
    print()
    
    print("=== REJECTED CONVERSATION ===")
    print(f"User: {first_prompt}")
    print(f"Assistant: {first_responses['llama_response']}")
    print()
    
    # Tokenize both conversations
    chosen_tokens = tokenizer.apply_chat_template(
        chosen_conversation,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )
    
    rejected_tokens = tokenizer.apply_chat_template(
        rejected_conversation,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )
    
    print("=== TOKENIZATION RESULTS ===")
    print(f"Chosen tokens shape: {chosen_tokens.shape}")
    print(f"Rejected tokens shape: {rejected_tokens.shape}")
    print()
    
    # Test our assistant response index logic
    assistant_token_id = 128007  # <|end_header_id|>
    
    def find_assistant_response_start(tokens):
        token_sequence = tokens[0]  # Remove batch dimension
        assistant_positions = (token_sequence == assistant_token_id).nonzero(as_tuple=True)[0]
        if assistant_positions.numel() == 0:
            raise ValueError("Assistant token not found in tokenized conversation.")
        # Find the last assistant header and start from the token AFTER it
        last_assistant_header_pos = assistant_positions[-1]
        assistant_response_start = last_assistant_header_pos + 1
        return assistant_response_start
    
    chosen_start = find_assistant_response_start(chosen_tokens)
    rejected_start = find_assistant_response_start(rejected_tokens)
    
    print("=== ASSISTANT RESPONSE INDICES ===")
    print(f"Chosen conversation length: {chosen_tokens.shape[1]}")
    print(f"Chosen assistant response starts at: {chosen_start}")
    print(f"Tokens from assistant start to end: {chosen_tokens.shape[1] - chosen_start}")
    print()
    print(f"Rejected conversation length: {rejected_tokens.shape[1]}")
    print(f"Rejected assistant response starts at: {rejected_start}")
    print(f"Tokens from assistant start to end: {rejected_tokens.shape[1] - rejected_start}")
    print()
    
    # Decode the assistant response portion to verify
    print("=== DECODED ASSISTANT RESPONSES ===")
    chosen_assistant_tokens = chosen_tokens[0, chosen_start:]
    rejected_assistant_tokens = rejected_tokens[0, rejected_start:]
    
    chosen_decoded = tokenizer.decode(chosen_assistant_tokens, skip_special_tokens=False)
    rejected_decoded = tokenizer.decode(rejected_assistant_tokens, skip_special_tokens=False)
    
    print("Chosen assistant response (decoded):")
    print(repr(chosen_decoded))
    print()
    print("Rejected assistant response (decoded):")
    print(repr(rejected_decoded))
    print()
    
    # Show some context around the assistant header
    print("=== CONTEXT AROUND ASSISTANT HEADER ===")
    chosen_context_start = max(0, chosen_start - 5)
    chosen_context_end = min(chosen_tokens.shape[1], chosen_start + 10)
    chosen_context = chosen_tokens[0, chosen_context_start:chosen_context_end]
    
    print(f"Chosen context tokens (positions {chosen_context_start}:{chosen_context_end}):")
    print(chosen_context.tolist())
    print("Decoded:")
    print(repr(tokenizer.decode(chosen_context, skip_special_tokens=False)))


if __name__ == "__main__":
    test_assistant_response_index() 