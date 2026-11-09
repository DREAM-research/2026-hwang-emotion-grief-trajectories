import sys
from pathlib import Path

import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'util'))
from voxelwise import load_voxels, read_clinical, read_maps, save_maps


if len(sys.argv) not in [5, 6] or '--help' in sys.argv:
    print(f'Usage: python {Path(sys.argv[0]).name} <clinical_csv> <map_csv> <mask> <output> [var_clin]')
    sys.exit(0 if '--help' in sys.argv else 1)

analysis_csv, map_csv, mask_file, output = sys.argv[1:5]
var_clin = sys.argv[5] if len(sys.argv) == 6 else 'score'
covars = ['Age', 'isF', 'Education', 'HAM_D', 'DSL']
var_time = 'tp_n'
min_tp = 3

# Select visits and standardize
df_all = read_clinical(analysis_csv)
df_all = df_all.loc[df_all[var_clin].notna(), :].copy()
df_all['n_visits'] = df_all.groupby('ID')['ID'].transform('size')
df_all = df_all.loc[df_all['n_visits'] >= min_tp, :].reset_index(drop=True)
for var in [var_clin, var_time] + covars:
    df_all[f'{var}_z'] = stats.zscore(df_all[var].to_numpy(dtype=float), nan_policy='omit')

baseline_vars = [f'{var_clin}_z', 'DSL_z', 'Age_z', 'HAM_D_z']
df_baseline = df_all.sort_values('DSL').groupby('ID')[baseline_vars].first()
df_all = df_all.join(df_baseline.add_suffix('_baseline'), on='ID')

# Load task activation maps
df_maps = read_maps(map_csv)
df_all = df_all.loc[df_all['ID'].isin(df_maps.index), :].copy()
ids = pd.Index(df_all['ID'].unique())
if len(ids) <= len(covars) + 2:
    sys.exit('Too few participants with clinical and imaging data.')

voxel_data, mask_bool, affine = load_voxels(df_maps, ids, map_csv, mask_file, np.float16)
visit_rows = ids.get_indexer(df_all['ID'])
print(f'N = {len(ids)}')

# Fit voxelwise LME
covars_z = ['Age_z_baseline', 'isF_z', 'Education_z', 'HAM_D_z_baseline', 'DSL_z_baseline']
lm_formula = f"{var_clin}_z ~ {var_time}_z * v_z + {var_clin}_z_baseline + {' + '.join(covars_z)}"
re_formula = f'~1 + {var_time}_z'
term = f'{var_time}_z:v_z'
p = np.full(voxel_data.shape[1], np.nan, dtype=np.float32)
t = np.full(voxel_data.shape[1], np.nan, dtype=np.float32)

for i, v in enumerate(voxel_data.T):
    values = v[visit_rows].astype(float)
    if np.ptp(values) == 0:
        p[i], t[i] = 1, 0
        continue

    df_all['v_z'] = stats.zscore(values)
    try:
        mdl = smf.mixedlm(lm_formula, df_all, groups='ID', re_formula=re_formula).fit(
            reml=False, method='lbfgs', maxiter=1000, disp=False)
        p[i], t[i] = mdl.pvalues[term], mdl.tvalues[term]
    except (ValueError, np.linalg.LinAlgError):
        continue

save_maps(output, mask_bool, affine, np.round(t.astype(float), 5), np.round(p.astype(float), 5))
print(f'Missing fits: {(~np.isfinite(p)).sum()}')
print('Saved to', output)
