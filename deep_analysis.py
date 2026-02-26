import pandas as pd, numpy as np
from scipy.stats import spearmanr

raw = pd.read_csv('data.csv')
new_columns = [
    'cell_id', 'initial_label', 'initial_voltage', 'initial_impedance',
    'v42_label', 'v42_voltage', 'v42_impedance',
    'v25_label', 'v25_voltage', 'v25_impedance',
    'v36_label', 'v36_voltage', 'v36_empty', 'v36_impedance', 'capacity'
]
raw.columns = new_columns
drop_cols = ['initial_label', 'v42_label', 'v25_label', 'v36_label', 'v36_empty']
df = raw.drop(columns=drop_cols)
for col in [c for c in df.columns if c != 'cell_id']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# v36_impedance outlier check
print('=== v36_impedance outlier ===')
extreme = df[df['v36_impedance'] > 100]
print(f'Extreme count: {len(extreme)}')
if len(extreme) > 0:
    print(extreme[['cell_id', 'v36_impedance', 'capacity']])

# capacity range
print('\n=== Capacity range analysis ===')
cap = df['capacity']
print(f'Range: {cap.max() - cap.min():.4f} Ah')
print(f'CV: {cap.std() / cap.mean() * 100:.2f}%')
print(f'Skew: {cap.skew():.4f}, Kurt: {cap.kurtosis():.4f}')

# Remove extreme outlier for clean analysis
df_clean = df[df['v36_impedance'] < 100].copy()
print(f'\nClean data: {len(df_clean)} rows')

# Spearman vs Pearson
feats = ['initial_voltage', 'initial_impedance', 'v42_voltage', 'v42_impedance',
         'v25_voltage', 'v25_impedance', 'v36_voltage', 'v36_impedance']
print('\n=== Spearman vs Pearson (non-linearity) ===')
for f in feats:
    p = df_clean[f].corr(df_clean['capacity'])
    s, _ = spearmanr(df_clean[f], df_clean['capacity'])
    diff = abs(s) - abs(p)
    flag = ' *** NON-LINEAR' if abs(diff) > 0.05 else ''
    print(f'  {f:25s}: Pearson={p:+.4f}, Spearman={s:+.4f}, diff={diff:+.4f}{flag}')

# Polynomial features check
print('\n=== Polynomial features (squared) with capacity ===')
for f in feats:
    sq = df_clean[f] ** 2
    r = sq.corr(df_clean['capacity'])
    r_orig = df_clean[f].corr(df_clean['capacity'])
    if abs(r) > abs(r_orig) + 0.02:
        print(f'  {f}^2: r={r:+.4f} (vs original {r_orig:+.4f}) IMPROVED')

# Log transform
print('\n=== Log transform ===')
for f in feats:
    if (df_clean[f] > 0).all():
        log_f = np.log(df_clean[f])
        r = log_f.corr(df_clean['capacity'])
        r_orig = df_clean[f].corr(df_clean['capacity'])
        if abs(r) > abs(r_orig) + 0.02:
            print(f'  log({f}): r={r:+.4f} (vs original {r_orig:+.4f}) IMPROVED')

# Interaction terms
print('\n=== Top interaction terms (|r| > 0.2) ===')
interactions = []
for i, f1 in enumerate(feats):
    for f2 in feats[i+1:]:
        # product
        prod = df_clean[f1] * df_clean[f2]
        r = prod.corr(df_clean['capacity'])
        interactions.append((f'{f1} * {f2}', r))
        # ratio
        ratio = df_clean[f1] / df_clean[f2].replace(0, np.nan)
        r2 = ratio.dropna().corr(df_clean.loc[ratio.dropna().index, 'capacity'])
        interactions.append((f'{f1} / {f2}', r2))
        # diff
        diff_val = df_clean[f1] - df_clean[f2]
        r3 = diff_val.corr(df_clean['capacity'])
        interactions.append((f'{f1} - {f2}', r3))

interactions.sort(key=lambda x: abs(x[1]), reverse=True)
for name, r in interactions[:20]:
    print(f'  {name:50s}: r={r:+.4f}')

# Check if capacity has subgroups (clustering)
print('\n=== Capacity distribution segments ===')
bins = [4.7, 4.9, 5.0, 5.05, 5.1]
for i in range(len(bins)-1):
    mask = (df_clean['capacity'] >= bins[i]) & (df_clean['capacity'] < bins[i+1])
    print(f'  [{bins[i]:.2f}, {bins[i+1]:.2f}): {mask.sum()} cells ({mask.sum()/len(df_clean)*100:.1f}%)')

# Check installed packages
print('\n=== Installed packages ===')
import importlib
for pkg in ['autogluon.tabular', 'catboost', 'flaml', 'shap', 'xgboost', 'lightgbm', 'sklearn']:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', 'ok')
        print(f'  {pkg}: {v}')
    except ImportError:
        print(f'  {pkg}: NOT INSTALLED')
