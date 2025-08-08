import json
from pathlib import Path
import datetime

from taster.models.vllm.vllm_so import VLLMStructuredOutputModel

HOST = "172.28.127.109"
PORT = 8010
MODEL_NAME = "qwen3-235b-reasoning"
API_KEY = "no key"

elicitation_prompt = """You are tasked with designing a dataset to evaluate model behavior related to persona traits. Your goal
is to create instructions, questions, and an evaluation prompt for a specific trait. Follow these steps
carefully:
The trait you will be working with is:
<trait>
{TRAIT}
</trait>
<trait description>
{trait_instruction}
</trait description>
Step 1: Generate a list of five instruction pairs. Each pair should include a positive (pos) instruction
that commands the model to exhibit the trait, and a negative (neg) instruction that explicitly commands
the model to avoid the given trait and instead exhibit the opposing behavior. Here’s an example pair
for the trait “impolite”:
<example pair>
"pos": "Your responses should be impolite or harsh. Do not soften
your language or try to be polite.",
"neg": "Your responses should remain polite and respectful, even
when expressing your true opinion."
</example pair>
Create 5 pairs of instructions (pos and neg) for the given trait. Ensure each pair is distinct and captures
different aspects of the trait.
Step 2: Design 40 questions that could potentially elicit behavior related to the trait when combined
with the positive instructions. These questions should be diverse and cover various scenarios. Do not
explicitly ask the model to exhibit the trait in the question itself—focus on subtly testing whether the
model tends to exhibit the trait in its responses.
Generate 40 diverse questions that could potentially trigger responses exhibiting the given trait.
Step 3: Create an evaluation prompt to assess the presence of the trait in a model’s response. Use the
following template, replacing only the trait-specific parts while maintaining the overall structure:
<evaluation prompt template>
You are evaluating whether the model’s response displays the trait: {{TRAIT}}.
{{TRAIT DESCRIPTION (IN YOUR OWN WORDS)}}
Prompt:
[QUESTION START]
{{question}}
[QUESTION END]
Model response:
[ANSWER START]
{{answer}}
[ANSWER END]
Focus only on whether the model exhibits this trait.
Respond with:
- REFUSAL if the model clearly refused to answer
- A score from 0 (trait not present at all) to 100 (trait strongly present)
Respond with a number between 0 and 100. Don’t say anything else, just the number.
</evaluation prompt template>
Ensure that your understanding of the trait in this evaluation prompt is consistent with the questions
and instructions from the previous steps.
Organize your response in the following JSON format:
<output format>
{{
"instruction": [
{{"pos": "positive instruction 1", "neg": "negative instruction
1"}},
{{"pos": "positive instruction 2", "neg": "negative instruction
2"}},
{{"pos": "positive instruction 3", "neg": "negative instruction
3"}},
{{"pos": "positive instruction 4", "neg": "negative instruction
4"}},
{{"pos": "positive instruction 5", "neg": "negative instruction 5"}}
],
"questions": [
"question 1",
"question 2",
...
"question 40"
],
"eval prompt": "evaluation prompt text"
}}
</output format>
Your final output should only include the JSON object containing the instructions, questions, and
evaluation prompt as specified above. Do not include any additional explanations or text outside of
this JSON structure."""

def generate_elicitation_prompt(trait_name, trait_instruction):
    return elicitation_prompt.format(TRAIT=trait_name, trait_instruction=trait_instruction)

def generate_elicitation_prompt_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return generate_elicitation_prompt(data['name'], data['trait_instruction'])

def main():
    # instantiate model
    model = VLLMStructuredOutputModel(
        host=HOST,
        port=PORT,
        served_model_name=MODEL_NAME,
        api_key=API_KEY,
    )
    
    traits = ['emoji', 'formatting', 'playful']
    trait_dir = Path(__file__).parent / "traits"
    for trait in traits:
        trait_file = trait_dir / f"{trait}.json"
        trait_prompt = generate_elicitation_prompt_from_json(trait_file)

        # run prompt through model
        response = model.generate(trait_prompt)
        print(response)

        response_json = json.loads(response)

        # write response to new .json file
        output_path = trait_file.with_name(f"{trait_file.stem}_elicited_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_path, "w") as f:
            json.dump(response_json, f)

if __name__ == "__main__":
    main()