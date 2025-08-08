import json

all_results = "/mnt/vast/home/lawrence/steer-llama/steer_llama/traits/emoji_elicited_results.json"
filtered = "/mnt/vast/home/lawrence/steer-llama/steer_llama/traits/emoji_elicited_filtered.json"

all_results_json = json.load(open(all_results))
filtered_json = json.load(open(filtered))

# for each prompt in filtered_json, create a dictionary with the prompt as the key and the pi response and llama response values
filtered_results = {}
for prompt in filtered_json:
    filtered_results[prompt] = {
        "pi_response": all_results_json[prompt]["pi_response"],
        "llama_response": all_results_json[prompt]["llama_response"]
    }

# write the filtered results to a file
with open(filtered, "w") as f:
    json.dump(filtered_results, f)