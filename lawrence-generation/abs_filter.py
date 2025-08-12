import json

emoji_path = "/mnt/vast/home/lawrence/steer-llama/steer_llama/traits/elicited/emoji_elicited_results.json"
playful_path = "/mnt/vast/home/lawrence/steer-llama/steer_llama/traits/elicited/playful_elicited_results.json"
formatting_path = "/mnt/vast/home/lawrence/steer-llama/steer_llama/traits/elicited/formatting_elicited_results.json"

# load the results
with open(emoji_path, "r") as f:
    emoji_results = json.load(f)

with open(playful_path, "r") as f:
    playful_results = json.load(f)

with open(formatting_path, "r") as f:
    formatting_results = json.load(f)

# filter the results
emoji_filtered = []
for question in emoji_results:
    if abs(emoji_results[question]["score_difference"]) > 40:
        emoji_filtered.append(question)

playful_filtered = []
for question in playful_results:
    if abs(playful_results[question]["score_difference"]) > 40:
        playful_filtered.append(question)

formatting_filtered = []
for question in formatting_results:
    if abs(formatting_results[question]["score_difference"]) > 40:
        formatting_filtered.append(question)

# write the filtered results to files
with open(playful_path.replace("results", "filtered"), "w") as f:
    json.dump(emoji_filtered, f)

with open(playful_path.replace("results", "filtered"), "w") as f:
    json.dump(playful_filtered, f)

with open(formatting_path.replace("results", "filtered"), "w") as f:
    json.dump(formatting_filtered, f)



