import json,time,torch
from transformers import AutoTokenizer, AutoModelForCausalLM



def local_phi(model, messages,temperature):
    for _ in range(10):
        try:
            if "lora" in model:
                model_id = "PATH_TO_FINETUNED_PHI4"
            else:
                model_id = "microsoft/phi-4"
            
            if not 'model_hg' in globals():
                global tokenizer
                global model_hg
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model_hg = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model_hg.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model_hg.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            return tokenizer.decode(response, skip_special_tokens=True)
        except Exception as e:
                print("Error in generate: ", e)
                time.sleep(1)
        return "I cannot respond"

def local_llama(model, messages,temperature):
    for _ in range(10):
        try:
            if "lora" in model:
                model_id = "PATH_TO_FINETUNED_LLAMA3"
            else:
                model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

            if not 'model_hg' in globals():
                global tokenizer
                global model_hg
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model_hg = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model_hg.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model_hg.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            return tokenizer.decode(response, skip_special_tokens=True)
        except Exception as e:
                print("Error in generate: ", e)
                time.sleep(1)
        return "I cannot respond"


def generate(messages, client, model, json_format=False, temperature=0.7):
    if "llama-3-8b-instruct" in model:# and json_format:
        response = local_llama(model, messages, temperature)
        return response
    elif "phi" in model:# and json_format:
        assert not json_format
        response = local_phi(model, messages, temperature)
        return response
    elif "claude" in model:
        messages_nosys = [message for message in messages if message['role']!= 'system']
        response = client.messages.create(
        model=model,
        system=[message["content"] for message in messages if message['role']== 'system'][0],
        max_tokens=8192,
        messages=messages_nosys,
        temperature=temperature,
        )
        return response.content[0].text
    elif "llama" in model or "deepseek" in model:
        # use llama-api
        for ind_ in range(100):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    )
                return response.choices[0].message.content
            except Exception as e:
                print("Error in generate: ", e)
                print("generate: ", response.choices[0].message.content)
        return {}
    else:
        # openai gpt 
        for ind_ in range(10):
            try:
                if 'o1' in model or 'o3' in model:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages
                    )
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        response_format={"type": "text"} if not json_format else {"type": "json_object"}
                    )

                if json_format:
                    return json.loads(response.choices[0].message.content)

                return response.choices[0].message.content
            except Exception as e:
                print("Error in generate: ", e)
                print("generate: ", response.choices[0].message.content)
                time.sleep(2*ind_+1)
        return {}
