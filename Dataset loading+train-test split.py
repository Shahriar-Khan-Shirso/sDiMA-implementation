import pandas as pd
import numpy as np
from tqdm import tqdm

merged = pd.read_csv('merged_protein_data.csv')
merged['publicationYear'] = pd.to_numeric(merged['publicationYear'], errors='coerce')
merged = merged.drop_duplicates(subset=['sequence'])

train_raw = merged[merged['publicationYear'].between(2010, 2015)].copy()
test_raw  = merged[merged['publicationYear'] == 2016].copy()

print(f'Merged loaded   : {len(merged):,} rows')
print(f'Train 2010-2015 : {len(train_raw):,} rows')
print(f'Test  2016      : {len(test_raw):,} rows')
