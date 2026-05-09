SELECTION_RULES = [

    (0.52, 0.70),  
    (0.50, 0.65),   
    (0.47, 0.4),   
    (0.44, 0.35),  
    (0.41, 0.30),   
    (0.30, 0.28),
    (0.20, 0.10),
    (0.10, 0.05),
    (0.00, 0.00)
]

NOISE_SCORE_FLOOR = 0.30   

# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np

scored_df    = pd.read_csv('scored_sequences.csv')
cluster_stats = pd.read_csv('cluster_stats.csv')


avg_lookup = dict(zip(cluster_stats['cluster'], cluster_stats['avg_score']))

def get_pct_for_cluster(avg_score):
    """Return top-% to keep based on the cluster's average score."""
    for threshold, pct in SELECTION_RULES:
        if avg_score >= threshold:
            return pct
    return 0.0

selected_parts = []

real_clusters = sorted([c for c in avg_lookup if c != -1])

print('Selection per cluster:')
print(f'{"Cluster":>8}  {"Avg Score":>10}  {"Rule (%)":>9}  {"Kept":>6}  {"Of":>6}')
print('-' * 48)

for cid in real_clusters:
    avg   = avg_lookup[cid]
    pct   = get_pct_for_cluster(avg)
    chunk = scored_df[scored_df['cluster'] == cid].copy()

    if pct <= 0.0:
        print(f'{cid:>8}  {avg:>10.4f}  {"SKIP":>9}  {"0":>6}  {len(chunk):>6}')
        continue

    chunk_sorted = chunk.sort_values('combined_score', ascending=False)
    n_keep       = max(1, int(len(chunk_sorted) * pct))
    kept         = chunk_sorted.head(n_keep)

    selected_parts.append(kept)
    print(f'{cid:>8}  {avg:>10.4f}  {pct*100:>8.0f}%  {n_keep:>6}  {len(chunk):>6}')


noise_df = scored_df[scored_df['cluster'] == -1].copy()
if len(noise_df) > 0 and NOISE_SCORE_FLOOR < 1.0:
    noise_kept = noise_df[noise_df['combined_score'] >= NOISE_SCORE_FLOOR]
    print(f'\n  Noise: kept {len(noise_kept)} / {len(noise_df)} '
          f'(score ≥ {NOISE_SCORE_FLOOR})')
    if len(noise_kept) > 0:
        selected_parts.append(noise_kept)


if selected_parts:
    selected_df = pd.concat(selected_parts, ignore_index=True)
    selected_df = selected_df.sort_values('combined_score', ascending=False)
else:
    selected_df = pd.DataFrame(columns=scored_df.columns)

print(f'\n[OK] Total selected: {len(selected_df)} sequences')
print(f'     combined_score | mean: {selected_df["combined_score"].mean():.4f} | '
      f'min: {selected_df["combined_score"].min():.4f} | '
      f'max: {selected_df["combined_score"].max():.4f}')


selected_df.to_csv('selected_sequences.csv', index=False)
print(f'[OK] Saved → selected_sequences.csv')
