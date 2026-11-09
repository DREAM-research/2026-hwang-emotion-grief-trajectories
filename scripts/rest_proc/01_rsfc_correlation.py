import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'util'))
from voxelwise import load_voxels, read_clinical, read_maps, save_maps


if len(sys.argv) not in [5, 6] or '--help' in sys.argv:
    print(f'Usage: python {Path(sys.argv[0]).name} <clinical_csv> <map_csv> <mask> <output> [var_clin]')
    sys.exit(0 if '--help' in sys.argv else 1)

analysis_csv, map_csv, mask_file, output = sys.argv[1:5]
var_clin = sys.argv[5] if len(sys.argv) == 6 else 'score'
covars = ['Age', 'isF', 'Education', 'HAM_D', 'DSL']

# Baseline sample
df = read_clinical(analysis_csv)
df_maps = read_maps(map_csv)
df = df.loc[(df['tp_n'] == 0) & df['ID'].isin(df_maps.index), :]
df = df.dropna(subset=[var_clin] + covars).reset_index(drop=True)
if len(df) <= len(covars) + 2:
    sys.exit('Too few participants with clinical and imaging data.')

# Load resting-state seed maps
voxel_data, mask_bool, affine = load_voxels(df_maps, df['ID'], map_csv, mask_file, np.float64)

# Correlate covariate residuals
covars_z = stats.zscore(df[covars].to_numpy())
dv = stats.zscore(df[var_clin].to_numpy(dtype=float))
dv_res = dv - covars_z @ np.linalg.lstsq(covars_z, dv, rcond=None)[0]
iv_res = voxel_data - covars_z @ np.linalg.lstsq(covars_z, voxel_data, rcond=None)[0]

r = np.zeros(voxel_data.shape[1])
p = np.ones(voxel_data.shape[1])
for i, iv in enumerate(iv_res.T):
    if np.ptp(iv) > 0:
        r[i], p[i] = stats.pearsonr(iv, dv_res)

save_maps(output, mask_bool, affine, r, p)
print(f'N = {len(df)}')
print('Saved to', output)
