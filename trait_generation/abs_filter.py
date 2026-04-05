import json
from pathlib import Path


ELICITED_DIR = Path(__file__).parent / "traits" / "elicited"
RESULT_FILES = {
    "emoji": ELICITED_DIR / "emoji_elicited_results.json",
    "playful": ELICITED_DIR / "playful_elicited_results.json",
    "formatting": ELICITED_DIR / "formatting_elicited_results.json",
}
THRESHOLD = 40


for trait, result_path in RESULT_FILES.items():
    with result_path.open() as f:
        trait_results = json.load(f)

    filtered_questions = [
        question
        for question, scores in trait_results.items()
        if abs(scores["score_difference"]) > THRESHOLD
    ]

    output_path = result_path.with_name(f"{trait}_elicited_filtered.json")
    with output_path.open("w") as f:
        json.dump(filtered_questions, f)


