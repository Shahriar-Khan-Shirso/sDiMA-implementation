
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import Counter

try:
    import hdbscan
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'hdbscan', '-q'])
    import hdbscan

try:
    import umap
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'umap-learn', '-q'])
    import umap

scored_df = pd.read_csv('scored_sequences.csv')
seqs      = scored_df['sequence'].tolist()
print(f'[OK] Loaded {len(seqs)} scored sequences')


print('\n[1] Computing ESM-2 mean-pooled embeddings...')
encoder.eval()
EMB_BATCH = 16
all_embs  = []

with torch.no_grad():
    for i in tqdm(range(0, len(seqs), EMB_BATCH)):
        batch    = seqs[i:i+EMB_BATCH]
        reps, mask = encoder.encode(batch)
        mask_f   = mask.unsqueeze(-1).float()
        summed   = (reps * mask_f).sum(dim=1)
        counts   = mask_f.sum(dim=1).clamp(min=1)
        all_embs.append((summed / counts).cpu().numpy())

emb_matrix = np.vstack(all_embs)


norms      = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
emb_matrix = emb_matrix / (norms + 1e-8)
print(f'[OK] Embedding matrix: {emb_matrix.shape}')

print('\n[2] PCA whitening (480 → 50)...')
pca     = PCA(n_components=50, random_state=42)
emb_pca = pca.fit_transform(emb_matrix)
print(f'[OK] variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%')

print('\n[3] UMAP reduction (50 → 15)...')
reducer = umap.UMAP(
    n_components = 15,
    n_neighbors  = 15,    
    min_dist     = 0.0,    
    metric       = 'cosine',
    random_state = 42,
    low_memory   = False,
)
emb_umap = reducer.fit_transform(emb_pca)
print(f'[OK] UMAP done | shape: {emb_umap.shape}')


MIN_CLUSTER_SIZE = 30   
MIN_SAMPLES      = 5
TARGET_CLUSTERS  = 20  

print(f'\n[4] HDBSCAN on UMAP embedding '
      f'(min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES})...')

clusterer = hdbscan.HDBSCAN(
    min_cluster_size         = MIN_CLUSTER_SIZE,
    min_samples              = MIN_SAMPLES,
    cluster_selection_method = 'eom',
    metric                   = 'euclidean',  
    core_dist_n_jobs         = -1,
)
labels     = clusterer.fit_predict(emb_umap)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()

print(f'[OK] HDBSCAN done')
print(f'     Clusters : {n_clusters}  |  Noise: {n_noise} ({n_noise/len(labels)*100:.1f}%)')

size_counts = Counter(labels)
for cid in sorted(size_counts):
    tag = f'cluster {cid}' if cid != -1 else 'noise (-1)'
    print(f'     {tag:15s}: {size_counts[cid]}')

if n_clusters < TARGET_CLUSTERS:
    print(f'\n[!] Only {n_clusters} HDBSCAN clusters — '
          f'falling back to K-Means (k={TARGET_CLUSTERS}) on UMAP embedding')

    km = MiniBatchKMeans(
        n_clusters   = TARGET_CLUSTERS,
        random_state = 42,
        n_init       = 10,
        batch_size   = 1024,
    )
    labels     = km.fit_predict(emb_umap)
    n_clusters = TARGET_CLUSTERS
    n_noise    = 0    
    METHOD     = 'kmeans'

    size_counts = Counter(labels)
    print(f'[OK] K-Means done | {n_clusters} clusters')
    for cid in sorted(size_counts):
        print(f'     cluster {cid:3d}: {size_counts[cid]} sequences')
else:
    METHOD = 'hdbscan'

print(f'\n[OK] Clustering method used : {METHOD}')
print(f'     Total clusters         : {n_clusters}')
print(f'     Noise points           : {n_noise}')

scored_df['cluster'] = labels
scored_df.to_csv('scored_sequences.csv', index=False)
np.save('seq_embeddings_umap.npy', emb_umap)  

print(f'\n[OK] cluster column added → scored_sequences.csv')
