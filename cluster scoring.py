
import pandas as pd
import numpy as np

scored_df = pd.read_csv('scored_sequences.csv')
print(f'[OK] Loaded {len(scored_df)} sequences | '
      f'{scored_df["cluster"].nunique()} unique labels (including -1 noise)')


cluster_stats = (
    scored_df
    .groupby('cluster')['combined_score']
    .agg(avg_score='mean', n='count', min_score='min', max_score='max')
    .reset_index()
    .sort_values('avg_score', ascending=False)
)

print('\nCluster averages (sorted best → worst):')
print(f'{"Cluster":>8}  {"Avg Score":>10}  {"Count":>7}  {"Min":>7}  {"Max":>7}')
print('-' * 50)
for _, row in cluster_stats.iterrows():
    label = f'{int(row.cluster):>4}' if row.cluster != -1 else 'noise'
    print(f'{label:>8}  {row.avg_score:>10.4f}  {int(row.n):>7}  '
          f'{row.min_score:>7.4f}  {row.max_score:>7.4f}')


real_clusters = cluster_stats[cluster_stats['cluster'] != -1]
print(f'\nSummary (noise excluded):')
print(f'  Clusters with avg ≥ 0.70 : {(real_clusters.avg_score >= 0.70).sum()}')
print(f'  Clusters with avg ≥ 0.60 : {(real_clusters.avg_score >= 0.60).sum()}')
print(f'  Clusters with avg ≥ 0.50 : {(real_clusters.avg_score >= 0.50).sum()}')
print(f'  Clusters with avg ≥ 0.40 : {(real_clusters.avg_score >= 0.40).sum()}')
print(f'  Clusters with avg ≥ 0.30 : {(real_clusters.avg_score >= 0.30).sum()}')
print(f'  Clusters with avg < 0.30 : {(real_clusters.avg_score < 0.30).sum()}')


cluster_stats.to_csv('cluster_stats.csv', index=False)
print(f'\n[OK] Saved → cluster_stats.csv')
