import json
from pathlib import Path
import os
import heapq
from tqdm import tqdm
import re

from taster.configs.pi_config import PiConfig  
from taster.models.pi.pi_llm import PiLLM
from taster.models.vllm.vllm import VLLMModel
from taster.models.vllm.vllm_so import VLLMStructuredOutputModel

PI_API_KEY = os.getenv("PI_API_KEY")
PI_API_URL = "https://api.inflection.ai/external/api/inference/openai/v1/chat/completions"
PI_SERVED_MODEL_NAME = "prod_fudge_6"

QWEN_API_KEY = "no key"
QWEN_API_URL = "http://172.28.127.109:8010/v1/chat/completions"
QWEN_SERVED_MODEL_NAME = "qwen3-235b-reasoning"

LLAMA_API_KEY = "no key"
LLAMA_API_URL = "http://172.28.127.202:8000/v1/chat/completions"
LLAMA_SERVED_MODEL_NAME = "llama-3.3-70b-instruct"

def main():
    pi_config = PiConfig(
            config=PI_SERVED_MODEL_NAME,
            temperature=0.8,
            max_tokens=1000,
            top_p=0.95,
            web_search=False,
            stop_tokens=None # does not work currently
        )
    pi_model = PiLLM(
        pi_config=pi_config,
        api_url=PI_API_URL,
        api_token=PI_API_KEY,
    )

    qwen_model = VLLMStructuredOutputModel(
        host=QWEN_API_URL,
        port=QWEN_API_KEY,
        served_model_name=QWEN_SERVED_MODEL_NAME,
        api_key=QWEN_API_KEY,
    )

    llama_model = VLLMModel(
        host="http://172.28.127.202:8000",
        port=8000,
        served_model_name=LLAMA_SERVED_MODEL_NAME,
        api_key=LLAMA_API_KEY,
    )

    trait_dir = Path(__file__).parent / "traits"
    for trait_file in trait_dir.glob("*_elicited.json"):
        with open(trait_file, "r") as f:
            trait_data = json.load(f)
            evaluation_prompt = trait_data["eval prompt"]
            questions = trait_data["questions"]

            results = {}
            for question in tqdm(questions):
                # ask pi for a response
                pi_response = pi_model.generate(question)

                # ask llama for a response
                llama_response = llama_model.generate(question)

                # ask qwen to judge the pi response using the eval prompt (0-100)
                qwen_judge_of_pi_raw = qwen_model.generate(evaluation_prompt.format(question=question, answer=pi_response))
                match_pi = re.search(r"\d+", str(qwen_judge_of_pi_raw))
                qwen_judge_of_pi = int(match_pi.group()) if match_pi else 0

                # ask qwen to judge the llama response using the eval prompt (0-100)
                qwen_judge_of_llama_raw = qwen_model.generate(evaluation_prompt.format(question=question, answer=llama_response))
                match_llama = re.search(r"\d+", str(qwen_judge_of_llama_raw))
                qwen_judge_of_llama = int(match_llama.group()) if match_llama else 0

                # take the difference in scores between pi and llama
                score_difference = qwen_judge_of_pi - qwen_judge_of_llama

                results[question] = {
                    "pi_response": pi_response,
                    "llama_response": llama_response,
                    "score_difference": score_difference
                }

        # write the results to a file
        with open(trait_file.with_name(f"{trait_file.stem}_results.json"), "w") as f:
            json.dump(results, f)

        # filter the traits by score difference; threshold is 40
        filtered_traits = []
        for trait in results:
            if results[trait]["score_difference"] > 40:
                filtered_traits.append(trait)
        
        # write the filtered traits to a file   
        with open(trait_file.with_name(f"{trait_file.stem}_filtered.json"), "w") as f:
            json.dump(filtered_traits, f)


if __name__ == "__main__":
    main()