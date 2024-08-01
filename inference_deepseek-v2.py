import re
import os
import csv
import json
import argparse
import pandas as pd

from openai import OpenAI

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

client = OpenAI(api_key="your_api_key", base_url="https://api.deepseek.com")

def main():
    # Prepare CSV file for output
    output_csv_file = f"./data/output/shuffle1/{dataset}_DeepSeek-V2_greedy.csv"

    # Load the prompts from the JSON file
    prompts_dict = load_prompts_from_json(data_path)

    fieldnames = ['Date', 'Response', 'Prediction']

    with open(output_csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:  # Check if file is empty to write header
            writer.writeheader()

        for date, prompt in prompts_dict.items():
            # shuffle1
            prompt = prompt.replace("Only output: Strongly Bullish, Bullish, Slightly Bullish, Flat, Fluctuating, Slightly Bearish, Bearish, Strongly Bearish", "Only output: Flat, Slightly Bearish, Bullish, Bearish, Fluctuating, Slightly Bullish, Strongly Bullish, Strongly Bearish")
            # # shuffle2
            # prompt = prompt.replace("Only output: Strongly Bullish, Bullish, Slightly Bullish, Flat, Fluctuating, Slightly Bearish, Bearish, Strongly Bearish", "Only output: Strongly Bearish, Flat, Slightly Bearish, Slightly Bullish, Fluctuating, Bullish, Bearish, Strongly Bullish")
            # # shuffle3
            # prompt = prompt.replace("Only output: Strongly Bullish, Bullish, Slightly Bullish, Flat, Fluctuating, Slightly Bearish, Bearish, Strongly Bearish", "Only output: Slightly Bearish, Strongly Bullish, Bearish, Bullish, Slightly Bullish, Fluctuating, Flat, Strongly Bearish")
            # # shuffle4
            # prompt = prompt.replace("Only output: Strongly Bullish, Bullish, Slightly Bullish, Flat, Fluctuating, Slightly Bearish, Bearish, Strongly Bearish", "Only output: Bearish, Strongly Bullish, Fluctuating, Slightly Bearish, Slightly Bullish, Strongly Bearish, Bullish, Flat")

            # prompt style
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0,
                stream=False
            )
            
            response = response.choices[0].message.content

            prediction = extract_prediction(response)
            
            # Write to CSV file
            writer.writerow({'Date': date, 'Response': response, 'Prediction': prediction})
            csvfile.flush()  # Flush data to disk after each write


if __name__ == "__main__":
    main()