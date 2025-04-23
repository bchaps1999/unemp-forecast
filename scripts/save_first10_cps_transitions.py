import pandas as pd

# 1. Define paths (relative to project root)
input_path  = 'data/processed/cps_transitions.csv'
output_path = 'data/processed/cps_transitions_head10.csv'

# 2. Read only first 10 rows and write
df = pd.read_csv(input_path, nrows=10)
df.to_csv(output_path, index=False)

print(f"Saved first 10 rows to {output_path}")
