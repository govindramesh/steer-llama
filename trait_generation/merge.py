import json
from pathlib import Path


ELICITED_DIR = Path(__file__).parent / "traits" / "elicited"
ALL_RESULTS_PATH = ELICITED_DIR / "playful_elicited_results.json"
FILTERED_PATH = ELICITED_DIR / "playful_elicited_filtered.json"


with ALL_RESULTS_PATH.open() as f:
    all_results_json = json.load(f)

with FILTERED_PATH.open() as f:
    filtered_json = json.load(f)

# For each selected prompt, keep the paired model responses only.
filtered_results = {}
for prompt in filtered_json:
    filtered_results[prompt] = {
        "pi_response": all_results_json[prompt]["pi_response"],
        "llama_response": all_results_json[prompt]["llama_response"],
    }

with FILTERED_PATH.open("w") as f:
    json.dump(filtered_results, f)
