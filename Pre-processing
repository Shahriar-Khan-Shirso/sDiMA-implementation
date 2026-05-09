
import pandas as pd
import numpy as np
from tqdm import tqdm


def filter_proteins(df):
    mask = df['macromoleculeType_x'].str.upper().str.contains('PROTEIN', na=False)
    return df[mask].copy()

train = filter_proteins(train_raw)
test  = filter_proteins(test_raw)
print(f'After protein filter -> train: {len(train):,} | test: {len(test):,}')


null_frac = train.isnull().mean()
drop_cols = null_frac[null_frac > 0.75].index.tolist()
train = train.drop(columns=drop_cols)
test  = test.drop(columns=[c for c in drop_cols if c in test.columns])
print(f'Dropped {len(drop_cols)} high-null cols (>80% null threshold)')


train = train.drop_duplicates(subset=['structureId', 'chainId'])
test  = test.drop_duplicates(subset=['structureId', 'chainId'])
print(f'After struct dedup -> train: {len(train):,} | test: {len(test):,}')


VALID_AA = set('ACDEFGHIKLMNPQRSTVWYX')

def clean_sequence(seq):
    if not isinstance(seq, str):
        return None
    seq = seq.upper().strip()
    if len(seq) < 40 or len(seq) > 512:   # 10 is ESM-2 safe minimum
        return None
    if set(seq) - VALID_AA:
        return None
    return seq

tqdm.pandas(desc='Cleaning train sequences')
train = train.copy()
train['sequence'] = train['sequence'].progress_apply(clean_sequence)

tqdm.pandas(desc='Cleaning test sequences')
test = test.copy()
test['sequence'] = test['sequence'].progress_apply(clean_sequence)

train = train.dropna(subset=['sequence'])
test  = test.dropna(subset=['sequence'])
print(f'After sequence clean -> train: {len(train):,} | test: {len(test):,}')


train['seq_len'] = train['sequence'].str.len()
test['seq_len']  = test['sequence'].str.len()
print(f'[OK] seq_len | train mean: {train["seq_len"].mean():.0f} | range: {train["seq_len"].min()}-{train["seq_len"].max()}')


num_cols = ['residueCount_x', 'resolution', 'phValue',
            'densityMatthews', 'densityPercentSol', 'crystallizationTempK']

train_medians = {}
for col in num_cols:
    if col in train.columns:
        train[col] = pd.to_numeric(train[col], errors='coerce')
        if col == 'phValue':            train[col] = train[col].clip(0, 14)
        if col == 'crystallizationTempK': train[col] = train[col].clip(200, 400)
        if col == 'resolution':         train[col] = train[col].clip(0.5, 10.0)
        med = train[col].median()
        train[col] = train[col].fillna(med)
        train_medians[col] = med

for col in num_cols:
    if col in test.columns:
        test[col] = pd.to_numeric(test[col], errors='coerce')
        if col == 'phValue':            test[col] = test[col].clip(0, 14)
        if col == 'crystallizationTempK': test[col] = test[col].clip(200, 400)
        if col == 'resolution':         test[col] = test[col].clip(0.5, 10.0)
        test[col] = test[col].fillna(train_medians.get(col, test[col].median()))


scale_cols = ['resolution', 'phValue']
train_scales = {}
for col in scale_cols:
    if col in train.columns:
        mn, mx = train[col].min(), train[col].max()
        train[col] = (train[col] - mn) / (mx - mn + 1e-8)
        train_scales[col] = (mn, mx)
for col in scale_cols:
    if col in test.columns:
        mn, mx = train_scales.get(col, (test[col].min(), test[col].max()))
        test[col] = (test[col] - mn) / (mx - mn + 1e-8)


if 'experimentalTechnique' in train.columns:
    train['technique_code'] = train['experimentalTechnique'].astype('category').cat.codes
    cat_map = dict(enumerate(train['experimentalTechnique'].astype('category').cat.categories))
    inv_map = {v: k for k, v in cat_map.items()}
    test['technique_code'] = test['experimentalTechnique'].map(inv_map).fillna(-1).astype(int)


if 'classification' in train.columns:
    top_classes = train['classification'].value_counts().nlargest(20).index
    train['classification_clean'] = train['classification'].where(
        train['classification'].isin(top_classes), 'OTHER')
    test['classification_clean'] = test['classification'].where(
        test['classification'].isin(top_classes), 'OTHER')
    class_map = {v: i for i, v in enumerate(
        train['classification_clean'].astype('category').cat.categories)}
    train['classification_code'] = train['classification_clean'].map(class_map)
    test['classification_code']  = test['classification_clean'].map(class_map).fillna(-1).astype(int)
    print(f'[OK] classification encoded | classes: {len(top_classes)} + OTHER')


print('\n' + '=' * 50)
print('PREPROCESSING DONE')
print(f"train shape : {train.shape}")
print(f"test shape  : {test.shape}")
print(f"train seqs  : {train['sequence'].notna().sum():,}")
print(f"seq len distribution (train): p10={np.percentile(train['seq_len'],10):.0f} | "
      f"p50={np.percentile(train['seq_len'],50):.0f} | p90={np.percentile(train['seq_len'],90):.0f}")
print('=' * 50)

