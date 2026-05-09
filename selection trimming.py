
import pandas as pd
import numpy as np

selected_df = pd.read_csv('selected_sequences.csv')
print(f'[OK] Loaded {len(selected_df)} selected sequences')

# Take top 2000 by combined_score
selected_df = (selected_df
               .sort_values('combined_score', ascending=False)
               .head(2000)
               .reset_index(drop=True))
print(f'[OK] Trimmed to top 2000 by combined_score')

# Final output — keep only the columns needed for folding
fold_df = selected_df[['sequence', 'combined_score', 'bio_score',
                        'plddt_proxy', 'cluster']].copy()
fold_df.index.name = 'idx'

fold_df.to_csv('fold_sequence.csv', index=True)

print(f'\n[OK] Saved → fold_sequence.csv')
print(f'     Rows    : {len(fold_df)}')
print(f'     Columns : {list(fold_df.columns)}')
print(f'\n     Score summary:')
print(f'       combined_score  mean = {fold_df["combined_score"].mean():.4f}')
print(f'       bio_score       mean = {fold_df["bio_score"].mean():.4f}')
print(f'       plddt_proxy     mean = {fold_df["plddt_proxy"].mean():.4f}')
print(f'\n     Cluster distribution:')
print(fold_df['cluster'].value_counts().sort_index().to_string())
print(f'\n[DONE] fold_sequence.csv is ready for ESMFold / pLDDT scoring.')
