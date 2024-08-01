import pandas as pd

datasets = ['macro_news', 'macro_indicator', 'firm_news']
models = ['Phi-3-medium', 'Yi-1.5-34B-Chat', 'Llama-3-70B', 'Qwen2-72B-Instruct', 'Mixtral-8x22B-Instruct-v0.1', ]

for model in models:
    for dataset in datasets:

        out1 = pd.read_csv(f'./data/output/{dataset}_{model}_greedy.csv')
        out1 = out1['Prediction'].to_list()

        out2 = pd.read_csv(f'./data/output/shuffle1/{dataset}_{model}_greedy.csv')
        out2 = out2['Prediction'].to_list()

        # Calculate the sum of absolute differences
        abs_diff_sum = sum(abs(a - b) for a, b in zip(out1, out2))

        # Calculate the total number of elements
        total_elements = len(out1)

        # Calculate the average absolute difference
        average_abs_diff = abs_diff_sum / total_elements

        print(f"Difference score for {model} on {dataset} is: {average_abs_diff}")