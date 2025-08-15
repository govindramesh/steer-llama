# SteerPi

This repository contains an automated pipeline for discovering vectors that capture certain behaviors of traits in language models. We used Llama-3.3-70B-Instruct.

This technique was inspired from [Persona Vectors](https://arxiv.org/pdf/2507.21509) by Chen et al.

## Method

### Automated Pipeline

1. Given a trait description, ask an LLM to generate a set of prompts that might elicit the trait from a model. The LLM is also asked to generate an evaluation prompt for Judge LLM.
2. Use the (Judge LLM + evaluation prompt) to score responses from Pi and Llama-3.3-70B-Instruct.
3. Prompts whose corresponding pair of responses were judged to not have large difference in trait elicitation are removed
$${\text{prompt} : \text{Judge}(\text{LLM}_1(\text{prompt}), \text{LLM}_2(\text{prompt})) \geq \text{threshold}}$$

### Activation Collection via Teacher Forcing in Llama

1. We use the TransformerLens library to hook Llama.
2. We take **Pi responses and teach force them into Llama**, collecting residual stream activations for each layer.
3. We take Base Llama resposnes and collect residual stream activations for each layer.

### Computing the Persona Vector

With the activations collected, we devised a few ways to comptue the presona vector.

* For the markdown persona vector, we formed per-layer steering vectors by averaging across non-alphabet response tokens. We also tried averaging across markdown response tokens.
* For the **emoji persona vector**, we formed per-layer steering vectors by averaging across emoji tokens in Pi's response.
  * Llama did not output any emojis in its responses to the selecetd set of prompts, so we averaged across all Llama response tokens.

## Experiments

With the emoji vector, we generated this graph showing steering effectiveness (judged by our LLM judge):

Steering Effectivenss Across Layers:
![Steering Effectivenss Across Layers](/results/emoji/steering_results.png)

Coherence Across Layers
![Coherence Across Layers](/results/emoji/coherence_by_layer_by_alpha.png)

### Example Steering:

Prompt: "It's so sunny today, I think I need some sunscreen!"

> **Llama**: Don't forget to apply that sunscreen before heading out. It's great that you're thinking about protecting your skin from the sun's strong UV rays. What's your plan for the day? Heading to the beach, going for a hike, or just enjoying some time outdoors?

> **Steered-Llama**: That's a great idea! ☀️ Protecting your skin from the sun's harmful rays is super important! 🌞 What kind of sunscreen are you looking for? Do you have a favorite brand or SPF level? 🤔 I can give you some tips on how to choose the right one for you! 😊

> **Pi**: ☀️🌞 Sounds like a bright and beautiful day! You're right about needing some sunscreen, it's always important to protect your skin from those UV rays. Remember to choose a broad-spectrum sunscreen with at least SPF 30, and don't forget to reapply every two hours if you're spending time outdoors! 

## Reproducability

`lawrence-generation` contains a series of Python script which taken in a simple JSON file containing trait name and trait description and outputs the list of prompts and an evaluation prompt. Taken directly from Chen et al.

`govind-hooks` contains a Python script to collect activations using HookedTransformer.

`lawrence-steer` contains a .ipynb notebook for loading the steering vector, generating responses using a hooked model, and running a sweep across layer and steering cofficients. The plot generated has trait expression score (0-100) from the LLM judge.