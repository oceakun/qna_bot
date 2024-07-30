import pandas as pd

input_csv_path = 'train.csv'
output_csv_path = 'first_100_records.csv'

df = pd.read_csv(input_csv_path)

df_first_100 = df.head(100)

df_first_100.to_csv(output_csv_path, index=False)

print(f"The first 100 records have been saved to '{output_csv_path}'")
