# The framework is modified based on https://github.com/AIM-Intelligence/Automated-Multi-Turn-Jailbreaks/blob/main/main.py

import os
import datetime
import argparse
import json
from dotenv import load_dotenv
from attacks.utils import generate
from attacks.crescendomation.run import run_crescendomation
from attacks.opposite_day.run import run_opposite_day
from attacks.actor_attack.run import run_actor_attack
from attacks.acronym.run import run_acronym
import anthropic
from openai import OpenAI


# Load the API keys from the .env file
load_dotenv()
clients = {}
# code modified from https://github.com/AI45Lab/ActorAttack/blob/master/utils.py
def get_env_variable(var_name):
    """Fetch environment variable or return None if not set."""
    return os.getenv(var_name)

def initialize_clients():
    """Dynamically initialize available clients based on environment variables."""
    try:
        gpt_api_key = get_env_variable('GPT_API_KEY')
        gpt_base_url = get_env_variable('BASE_URL_GPT')
        if gpt_api_key and gpt_base_url:
            clients['gpt'] = OpenAI(base_url=gpt_base_url, api_key=gpt_api_key)

        claude_api_key = get_env_variable('CLAUDE_API_KEY')
        if claude_api_key:
            clients['claude'] = anthropic.Anthropic(
                                api_key=claude_api_key,
                            )
        
        llama_api_key = get_env_variable('LLAMA_API_KEY')
        llama_base_url = get_env_variable('BASE_URL_LLAMA')
        if llama_api_key and llama_base_url:
            clients['llama'] = OpenAI(base_url=llama_base_url, api_key=llama_api_key)

        if not clients:
            print("No valid API credentials found. Exiting.")
            exit(1)

    except Exception as e:
        print(f"Error during client initialization: {e}")
        exit(1)

def get_client(model_name):
    """Select appropriate client based on the given model name."""
    if 'gpt' in model_name or 'o1'in model_name or 'o3'in model_name:
        client = clients.get('gpt') 
    elif 'claude' in model_name:
        client = clients.get('claude')

    elif 'llama' in model_name or 'deepseek' in model_name:
        client = clients.get('llama')
    else:
        raise ValueError(f"Unsupported or unknown model name: {model_name}")

    if not client:
        raise ValueError(f"{model_name} client is not available.")
    return client 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, help="The name of target (victim) model, e.g. gpt-3.5-turbo, claude-3-5-sonnet-20241022, llama3.1-70b, etc.", default="gpt-4o")
    parser.add_argument("--attacker_model", type=str, help="The name of attacker model (used to generate multi-turn attacks and judge safety score)", default="gpt-4o")
    parser.add_argument("--attack_method", type=str, help="The multi-turn jailbreaking attack method", default="crescendomation")
    parser.add_argument("--safety_filtering", action='store_true', help="Whether to apply filtering-based safety steering")
    parser.add_argument("--add_safety_index", action='store_true', help="Whether to save safety scores (safety index) in the result files")
    parser.add_argument("--model_path", type=str, help="Path to the  well-trained models (including 'ssm' and 'nbf')", default="")
    parser.add_argument("--threshold", type=float, help="Steering threshold (\eta>=0), safe if safety index < -threshold, the larger it is, the stronger filtering will be", default=0) 
    parser.add_argument("--start_from", type=int, help="The index to start from, in ./data/test/harmbench_tasks.json", default=0) 
    parser.add_argument("--save_suffix", type=str, help="Suffix of file name to save the results", default="")
    args = parser.parse_args()


    initialize_clients()

    target_client = get_client(args.target_model)
    target_model = args.target_model
    target_generate = lambda messages, **kwargs: generate(messages, client=target_client, model=target_model, **kwargs)

    attacker_client = get_client(args.attacker_model) 
    attacker_model = args.attacker_model
    attacker_generate = lambda messages, **kwargs: generate(messages, client=attacker_client, model=attacker_model, **kwargs)

    with open('./data/test/harmbench_tasks.json', 'r') as f:
        test_case = json.load(f)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if args.safety_filtering:
        args.save_suffix += f"steering_eta{args.threshold}"
    output_file_path = f"./results/{args.attack_method}_{args.target_model}_{args.save_suffix}_{current_time}.jsonl"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    print("Generated Output file path:", output_file_path)


    for i,each_case in enumerate(test_case):
        if i < args.start_from: continue
        with open(output_file_path, 'a') as f:
            f.write(json.dumps({"Goal": each_case["task"]}) + '\n')
        if args.attack_method == "opposite_day":
            run_opposite_day(each_case, output_file_path, target_generate, attacker_generate,args)
        elif args.attack_method == "crescendomation":
            run_crescendomation(each_case, output_file_path, target_generate, attacker_generate,args)
        elif args.attack_method == "actor_attack":
            run_actor_attack(each_case, output_file_path, target_generate, attacker_generate,args)
        elif args.attack_method == "acronym":
            run_acronym(each_case, output_file_path, target_generate, attacker_generate,args)
        else:
            raise ValueError(f"Invalid jailbreak tactic: {args.attack_method}")