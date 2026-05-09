

import pandas as pd
import numpy as np
import random

df = merged.copy()
]
df['publicationYear'] = pd.to_numeric(df['publicationYear'], errors='coerce')

]
def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j-1], prev[j], dp[j-1])
    return dp[n]

def impact_change(s1, s2):
    max_len = max(len(s1), len(s2))
    total = 0
    for i in range(max_len):
        c1 = s1[i] if i < len(s1) else None
        c2 = s2[i] if i < len(s2) else None
        if c1 is None and c2 is None:
            total += 1
        elif c1 is None or c2 is None:
            total += 3
        else:
            total += abs(ord(c1) - ord(c2))
    return total / max_len

def hamming_padded(s1, s2):
    max_len = max(len(s1), len(s2))
    s1p = s1.ljust(max_len)
    s2p = s2.ljust(max_len)
    return sum(c1 != c2 for c1, c2 in zip(s1p, s2p)) / max_len

def pick_nearest_pairs(seqs_a, seqs_b, n):
    seqs_a = random.sample(seqs_a, min(350, len(seqs_a)))
    seqs_b = random.sample(seqs_b, min(350, len(seqs_b)))
    pairs = []
    used_b = set()
    candidates = [(abs(len(a)-len(b)), i, j, a, b)
                  for i, a in enumerate(seqs_a)
                  for j, b in enumerate(seqs_b)]
    candidates.sort()
    for _, i, j, a, b in candidates:
        if j not in used_b and len(pairs) < n:
            pairs.append((a, b))
            used_b.add(j)
        if len(pairs) == n:
            break
    return pairs

def ascii_to_step_desc(score):
    if score < 1:
        return "< 1 step (near-identical, e.g. A→B)"
    elif score < 2:
        return f"~{score:.1f} step change (e.g. A→C)"
    elif score < 4:
        return f"~{score:.1f} step change (e.g. A→E)"
    elif score < 7:
        return f"~{score:.1f} step change (e.g. A→H)"
    else:
        return f"~{score:.1f} step change "

YEAR_PAIRS = [(y, y+1) for y in range(2010, 2016)]

# ════════════════════════════════════════════════════════════════════════════════
# 1. LEVENSHTEIN
# ════════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. LEVENSHTEIN DISTANCE (10 pairs per consecutive year)")
print("=" * 60)

lev_year_avgs = []
lev_year_avg_lens = []
for y1, y2 in YEAR_PAIRS:
    seqs_a = df[df['publicationYear'] == y1]['sequence'].dropna().tolist()
    seqs_b = df[df['publicationYear'] == y2]['sequence'].dropna().tolist()
    if len(seqs_a) < 1 or len(seqs_b) < 1:
        print(f"  {y1}-{y2}: not enough data")
        continue
    pairs = pick_nearest_pairs(seqs_a, seqs_b, 10)
    dists = [levenshtein(a, b) for a, b in pairs]
    avg = np.mean(dists)
    avg_len = np.mean([len(a) for a, b in pairs] + [len(b) for a, b in pairs])
    pct = (avg / avg_len) * 100
    lev_year_avgs.append(avg)
    lev_year_avg_lens.append(avg_len)
    print(f"  {y1}-{y2}  →  {avg:.2f} edit distance")
    print(f"{avg:.2f} character changes out of avg {avg_len:.2f} length  |  {pct:.1f}% of sequence changed\n")

overall_lev = np.mean(lev_year_avgs)
overall_len = np.mean(lev_year_avg_lens)
overall_pct = (overall_lev / overall_len) * 100
print(f"  ★ Average across all year pairs: {overall_lev:.2f} edit distance")
print(f"{overall_lev:.2f} character changes out of avg {overall_len:.2f} length  |  {overall_pct:.1f}% of sequence changed on average")

# ════════════════════════════════════════════════════════════════════════════════
# 2. ASCII IMPACT CHANGE
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. ASCII IMPACT CHANGE (same technique & classification)")
print("=" * 60)

ascii_year_avgs = []
for y1, y2 in YEAR_PAIRS:
    sub_a = df[df['publicationYear'] == y1][['sequence','experimentalTechnique','classification']].dropna()
    sub_b = df[df['publicationYear'] == y2][['sequence','experimentalTechnique','classification']].dropna()
    sub_a = sub_a.groupby(['experimentalTechnique','classification'], group_keys=False).apply(
        lambda x: x.sample(min(len(x), 50)))
    sub_b = sub_b.groupby(['experimentalTechnique','classification'], group_keys=False).apply(
        lambda x: x.sample(min(len(x), 50)))
    ascii_merged = sub_a.merge(sub_b, on=['experimentalTechnique','classification'], suffixes=('_a','_b'))  # renamed to ascii_merged — avoids overwriting 'merged'
    if ascii_merged.empty:
        print(f"  {y1}-{y2}: no matching attribute pairs found\n")
        continue
    ascii_merged['len_diff'] = abs(ascii_merged['sequence_a'].str.len() - ascii_merged['sequence_b'].str.len())
    ascii_merged = ascii_merged.sort_values('len_diff').head(10)
    scores = [impact_change(r['sequence_a'], r['sequence_b']) for _, r in ascii_merged.iterrows()]
    avg = np.mean(scores)
    ascii_year_avgs.append(avg)
    print(f"  {y1}-{y2}  →  avg ASCII impact = {avg:.4f}")
    print(f"           {ascii_to_step_desc(avg)}\n")

overall_ascii = np.mean(ascii_year_avgs)
print(f"  ★ Average across all year pairs: {overall_ascii:.4f}")
print(f"{ascii_to_step_desc(overall_ascii)}")
print(f"(score = avg ASCII distance per position; a→e = 4 steps, a→b = 1 step)")

# ════════════════════════════════════════════════════════════════════════════════
# 3. HAMMING
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. HAMMING DISTANCE (padded, 3-4 pairs per consecutive year)")
print("=" * 60)

ham_year_avgs = []
ham_year_avg_lens = []  # cache lengths here — avoids re-running pick_nearest_pairs for final summary
for y1, y2 in YEAR_PAIRS:
    seqs_a = df[df['publicationYear'] == y1]['sequence'].dropna().tolist()
    seqs_b = df[df['publicationYear'] == y2]['sequence'].dropna().tolist()
    if len(seqs_a) < 1 or len(seqs_b) < 1:
        print(f"  {y1}-{y2}: not enough data")
        continue
    pairs = pick_nearest_pairs(seqs_a, seqs_b, 4)
    dists = [hamming_padded(a, b) for a, b in pairs]
    avg = np.mean(dists)
    avg_len = np.mean([len(a) for a, b in pairs] + [len(b) for a, b in pairs])
    diff_chars = avg * avg_len
    ham_year_avgs.append(avg)
    ham_year_avg_lens.append(avg_len)
    print(f"  {y1}-{y2}  →  {avg*100:.1f}% mismatch")
    print(f"Approximately {diff_chars:.0f} characters are different (out of avg length {avg_len:.0f})\n")

overall_ham     = np.mean(ham_year_avgs)
overall_ham_len = np.mean(ham_year_avg_lens)  # uses cached values, consistent with per-year results
overall_diff_chars = overall_ham * overall_ham_len
print(f"  ★ Average across all year pairs: {overall_ham*100:.1f}% mismatch")
print(f"    Approximately {overall_diff_chars:.0f} characters are different on average (out of avg length {overall_ham_len:.0f})")
