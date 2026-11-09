from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def read_clinical(analysis_csv):
    df = pd.read_csv(analysis_csv, dtype={'ID': str, 'id': str}, low_memory=False)
    df = df.rename(columns={
        'id': 'ID', 'month': 'tp_n', 'age': 'Age', 'female': 'isF',
        'education': 'Education', 'hdrs17': 'HAM_D', 'days_since_loss': 'DSL',
        'HAM-D': 'HAM_D', 'Days Since Loss': 'DSL', 'PG-13-R': 'PG_13_R',
        'UCLA Loneliness': 'Loneliness', 'IES-R Intrusion': 'Intrusion',
    })

    if 'isF' not in df.columns and 'Sex' in df.columns:
        df['isF'] = df['Sex'].map({'M': 0, 'F': 1})
    if 'tp_n' not in df.columns and 'Timepoint' in df.columns:
        df['tp_n'] = df['Timepoint'].map({f'M{i:02d}': i for i in range(0, 13, 2)})
    if 'tp_n' not in df or not df['tp_n'].isin(range(0, 13, 2)).all():
        raise ValueError('Supply Timepoint as M00, M02, ..., M12 or tp_n as months.')
    if df.duplicated(['ID', 'tp_n']).any():
        raise ValueError('Duplicate clinical visits.')
    return df


def read_maps(map_csv):
    df_maps = pd.read_csv(map_csv, dtype={'ID': str, 'id': str})
    df_maps = df_maps.rename(columns={'id': 'ID'})
    return df_maps.set_index('ID', verify_integrity=True)


def load_voxels(df_maps, ids, map_csv, mask_file, dtype):
    mask_img = nib.load(mask_file)
    mask_bool = mask_img.get_fdata() != 0
    if mask_bool.ndim != 3 or not mask_bool.any():
        raise ValueError('Supply a nonempty 3D analysis mask.')

    voxel_data = np.empty((len(ids), int(mask_bool.sum())), dtype=dtype)
    for i, img_file in enumerate(df_maps.loc[ids, 'image']):
        img = nib.load(Path(map_csv).parent / img_file)
        if img.shape != mask_bool.shape or not np.allclose(img.affine, mask_img.affine):
            raise ValueError(f'Image grid does not match the mask: {img_file}')
        voxel_data[i, :] = img.get_fdata()[mask_bool]

    if not np.isfinite(voxel_data).all():
        raise ValueError('Nonfinite values inside the analysis mask.')
    return voxel_data, mask_bool, mask_img.affine


def save_maps(output, mask_bool, affine, stat, p):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    stat_img = np.zeros(mask_bool.shape)
    p_img = np.ones(mask_bool.shape)
    stat_img[mask_bool] = stat
    p_img[mask_bool] = p
    nib.save(nib.Nifti1Image(stat_img, affine), f'{output}_stat.nii.gz')
    nib.save(nib.Nifti1Image(p_img, affine), f'{output}_p.nii.gz')
