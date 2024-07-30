import pandas as pd

jsonl_file_path = 'simple_data.jsonl'
csv_file_path = 'simple_data.csv'

df = pd.read_json(jsonl_file_path, lines=True)

df.to_csv(csv_file_path, index=False)

print(f"Converted {jsonl_file_path} to {csv_file_path}")
