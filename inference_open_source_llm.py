import re
import os
import csv
import json
import argparse
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BertTokenizer, BertForSequenceClassification, pipeline, LlamaForCausalLM, LlamaTokenizerFast
import torch
from peft import PeftModel


# dataset = "macro_indicator"
dataset = "firm_news"
data_path = f"./data/prompt/{dataset}.json"

# Function to load JSON data
def load_prompts_from_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

# Function to extract prediction
def extract_prediction(text):
    prediction_mapping = {
        'Strongly Bullish': 3,
        'Bullish': 2,
        'Slightly Bullish': 1,
        'Flat': 0,
        'Fluctuating': 0,
        'Slightly Bearish': -1,
        'Bearish': -2,
        'Strongly Bearish': -3
    }
    
    match = re.search(r'\b(Strongly Bullish|Bullish|Slightly Bullish|Flat|Fluctuating|Slightly Bearish|Bearish|Strongly Bearish)\b', text, re.IGNORECASE)
    if match:
        return prediction_mapping.get(match.group(1))
    return None

def get_finbert_sentiment(probabilities):
    # positive, negative, neutral
    prediction = torch.argmax(probabilities).item()
    mapping = {0: 2, 1: -2, 2: 0}
    return mapping[prediction]

def get_input_ids_and_attention_mask_chunk(tokens):
    """
    This function splits the input_ids and attention_mask into chunks of size 'chunksize'. 
    It also adds special tokens (101 for [CLS] and 102 for [SEP]) at the start and end of each chunk.
    If the length of a chunk is less than 'chunksize', it pads the chunk with zeros at the end.
    
    Returns:
        input_id_chunks (List[torch.Tensor]): List of chunked input_ids.
        attention_mask_chunks (List[torch.Tensor]): List of chunked attention_masks.
    """
    chunksize = 512
    input_id_chunks = list(tokens['input_ids'][0].split(chunksize - 2))
    attention_mask_chunks = list(tokens['attention_mask'][0].split(chunksize - 2))
    
    for i in range(len(input_id_chunks)):
        input_id_chunks[i] = torch.cat([
            torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
        ])
        
        attention_mask_chunks[i] = torch.cat([
            torch.tensor([1]), attention_mask_chunks[i], torch.tensor([1])
        ])
        
        pad_length = chunksize - input_id_chunks[i].shape[0]
        
        if pad_length > 0:
            input_id_chunks[i] = torch.cat([
                input_id_chunks[i], torch.Tensor([0] * pad_length)
            ])
            attention_mask_chunks[i] = torch.cat([
                attention_mask_chunks[i], torch.Tensor([0] * pad_length)
            ])
            
    return input_id_chunks, attention_mask_chunks 
    

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Finance LLM")

    # Add arguments
    parser.add_argument('--model', type=str, default='Llama-3-70B', help='Model to use (default: Llama-3-70B)')

    # Parse the arguments
    args = parser.parse_args()
    
    # Print all the arguments
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    if args.model == "Llama-3-70B":
        model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    elif args.model == "Qwen2-72B-Instruct":
        model_name = "Qwen/Qwen2-72B-Instruct"
    elif args.model == "Mixtral-8x22B-Instruct-v0.1":
        model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    elif args.model == "Yi-1.5-34B-Chat":
        model_name = "01-ai/Yi-1.5-34B-Chat"
    elif args.model == "Phi-3-medium":
        model_name = "microsoft/Phi-3-medium-4k-instruct"
    elif args.model == "FinBERT":
        model_name = "ProsusAI/finbert"
    elif args.model == "FinGPT":
        base_model = "NousResearch/Llama-2-13b-hf" 
        peft_model = "FinGPT/fingpt-sentiment_llama2-13b_lora"

    if args.model == "Phi-3-medium":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
    elif args.model == "Qwen2-72B-Instruct":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
    elif args.model == "Yi-1.5-34B-Chat":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif args.model == "FinBERT":
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
    elif args.model == "FinGPT":
        tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="cuda:0")
        model = PeftModel.from_pretrained(model, peft_model)
        model = model.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    
    # Prepare CSV file for output
    output_csv_file = f"./data/output/shuffle1/{dataset}_{args.model}_greedy.csv"

    # Load the prompts from the JSON file
    prompts_dict = load_prompts_from_json(data_path)

    # print(f"Date: {date}\nPrompt: {prompt}\n")
    fieldnames = ['Date', 'Response', 'Prediction']

    with open(output_csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:  # Check if file is empty to write header
            writer.writeheader()
            
        for date, prompt in prompts_dict.items():\
            # shuffle1
            prompt = prompt.replace("Only output: Strongly Bullish, Bullish, Slightly Bullish, Flat, Fluctuating, Slightly Bearish, Bearish, Strongly Bearish", "Only output: Flat, Slightly Bearish, Bullish, Bearish, Fluctuating, Slightly Bullish, Strongly Bullish, Strongly Bearish")
            # # shuffle2
            # prompt = prompt.replace("Only output: Strongly Bullish, Bullish, Slightly Bullish, Flat, Fluctuating, Slightly Bearish, Bearish, Strongly Bearish", "Only output: Strongly Bearish, Flat, Slightly Bearish, Slightly Bullish, Fluctuating, Bullish, Bearish, Strongly Bullish")
            # # shuffle3
            # prompt = prompt.replace("Only output: Strongly Bullish, Bullish, Slightly Bullish, Flat, Fluctuating, Slightly Bearish, Bearish, Strongly Bearish", "Only output: Slightly Bearish, Strongly Bullish, Bearish, Bullish, Slightly Bullish, Fluctuating, Flat, Strongly Bearish")
            # # shuffle4
            # prompt = prompt.replace("Only output: Strongly Bullish, Bullish, Slightly Bullish, Flat, Fluctuating, Slightly Bearish, Bearish, Strongly Bearish", "Only output: Bearish, Strongly Bullish, Fluctuating, Slightly Bearish, Slightly Bullish, Strongly Bearish, Bullish, Flat")
            

            # prompt style
            if args.model in ["Mixtral-8x22B-Instruct-v0.1", "Yi-1.5-34B-Chat", "Phi-3-medium"]:
                messages = [
                    {"role": "user", "content": prompt},
                ]
            elif args.model == "Qwen2-72B-Instruct":
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                ]


            # decoding setting
            if args.model == "Qwen2-72B-Instruct":
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_length=4096,
                    do_sample=False,
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            elif args.model == "Phi-3-medium":
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )
                
                generation_args = {
                    "max_new_tokens": 4096,
                    "return_full_text": False,
                    "do_sample": False,
                }
                
                output = pipe(messages, **generation_args)
                response = output[0]['generated_text']
            elif args.model == "FinBERT":
                # drop the instructions, only keep the data
                prompt = prompt.split('\n\n')[1]
                tokens = tokenizer.encode_plus(prompt, add_special_tokens=False, return_tensors = 'pt')
                input_id_chunks, attention_mask_chunks = get_input_ids_and_attention_mask_chunk(tokens)
                input_ids = torch.stack(input_id_chunks)
                attention_mask = torch.stack(attention_mask_chunks)

                input_dict = {
                    'input_ids' : input_ids.long(),
                    'attention_mask' : attention_mask.int()
                }

                outputs = model(**input_dict)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=-1)
                mean_probabilities = probabilities.mean(dim=0)
                response = mean_probabilities
            elif args.model == "FinGPT":
                def split_text(text, max_length):
                    tokens = tokenizer(text, return_tensors='pt')
                    input_ids = tokens['input_ids'][0]
                    chunks = []
                    for i in range(0, len(input_ids), max_length):
                        chunk = tokenizer.decode(input_ids[i:i+max_length], skip_special_tokens=True)
                        chunks.append(chunk)
                    return chunks

                def sentiment_to_score(sentiment):
                    if 'positive' in sentiment and 'negative' in sentiment:
                        return 0
                    elif 'positive' in sentiment:
                        return 2
                    elif 'negative' in sentiment:
                        return -2
                    elif 'neutral' in sentiment:
                        return 0
                    return None  # Default for unknown sentiments

                def get_finbert_sentiment(prompt):
                    prompt_text = 'Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}\nInput: ' + prompt + '\nAnswer: '
                    tokens = tokenizer(prompt_text, return_tensors='pt', padding=True, max_length=512, truncation=True)
                    tokens = {key: val.to('cuda:0') for key, val in tokens.items()}
                    output = model.generate(**tokens, do_sample=False, max_length=512)
                    response = tokenizer.decode(output[0], skip_special_tokens=True)
                    if "Answer: " in response:
                        sentiment = response.split("Answer: ")[1].strip()
                        return sentiment
                    else:
                        None

                prompt = prompt.split('\n\n')[1]
                # prompt = 'Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}\nInput: ' + prompt + '\nAnswer: '
            
                # # Generate results
                # tokens = tokenizer(prompt, return_tensors='pt', padding=True, max_length=4096)
                # tokens = {key: val.to('cuda:0') for key, val in tokens.items()}

                # output = model.generate(**tokens, max_length=4096)

                # # Decode results
                # response = tokenizer.decode(output[0], skip_special_tokens=True)
                # response = response.split("Answer: ")[1].strip()


                # FinGPT generates the garbled text when the number of tokens is greater than 512
                chunks = split_text(prompt, 470) # make sure prompt = (instruction + data) <= 500 max tokens
                sentiments = []
                scores = []

                for chunk in chunks:
                    sentiment = get_finbert_sentiment(chunk)
                    if sentiment != None:
                        sentiments.append(sentiment)
                        score = sentiment_to_score(sentiment)
                        if score != None:
                            scores.append(score)

                response = sentiments
                average_score = sum(scores) / len(scores)

            else:
                if args.model == "Yi-1.5-34B-Chat":
                    input_ids = tokenizer.apply_chat_template(
                        conversation=messages, 
                        tokenize=True, 
                        return_tensors='pt'
                    ).to(model.device)
                else:
                    input_ids = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(model.device)

                if args.model == "Llama-3-80B":
                    terminators = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    outputs = model.generate(
                        input_ids,
                        max_length=4096,
                        eos_token_id=terminators,
                        do_sample=False,
                        num_beams=1,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                else:
                    outputs = model.generate(
                        input_ids,
                        max_length=4096,
                        do_sample=False,
                        num_beams=1,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                response = outputs[0][input_ids.shape[-1]:]
                response = tokenizer.decode(response, skip_special_tokens=True)

            if args.model == "FinBERT":
                prediction = get_finbert_sentiment(response)
            elif args.model == "FinGPT":
                prediction = average_score
            else:
                prediction = extract_prediction(response)
            # Write to CSV file
            writer.writerow({'Date': date, 'Response': response, 'Prediction': prediction})
            csvfile.flush()  # Flush data to disk after each write


if __name__ == "__main__":
    main()