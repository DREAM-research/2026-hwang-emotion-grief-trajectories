"""Apply a study 3dClustSim NN1 two-sided extent table to one search mask."""

import argparse
import csv
import math
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure, label


def minimum_extent(path):
    side = nn = alpha = None
    extents = set()
    for line in path.read_text().splitlines():
        match = re.match(r'#\s+(1-sided|2-sided|bi-sided) thresholding', line)
        if match:
            side, nn, alpha = match[1], None, None
        match = re.match(r'#\s+-NN\s+(\d+)', line)
        if match:
            nn, alpha = int(match[1]), None
        if side != '2-sided' or nn != 1:
            continue
        if 'pthr' in line.lower() and '|' in line:
            alpha = [float(x) for x in line.split('|', 1)[1].split()]
        elif alpha and line.strip() and not line.lstrip().startswith('#'):
            p, *values = map(float, line.replace('|', ' ').split())
            if np.isclose(p, .005):
                column = next(i for i, a in enumerate(alpha) if np.isclose(a, .05))
                extents.add(math.ceil(values[column]))
    if len(extents) != 1:
        raise ValueError('Missing or conflicting NN1 two-sided p=.005, alpha=.05 thresholds.')
    return extents.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['cross-sectional', 'longitudinal'])
    parser.add_argument('statistics', type=Path)
    parser.add_argument('pvalues', type=Path)
    parser.add_argument('mask', type=Path)
    parser.add_argument('table', type=Path, help='Matching AFNI NN1 2sided table')
    parser.add_argument('output', type=Path, help='Output filename prefix')
    args = parser.parse_args()

    extent = minimum_extent(args.table)
    image, pimage, mask = [nib.load(p) for p in (args.statistics, args.pvalues, args.mask)]
    if any(img.shape != image.shape or not np.allclose(img.affine, image.affine)
           for img in (pimage, mask)):
        parser.error('The maps and mask must use the same grid.')
    values = image.get_fdata()
    inside = mask.get_fdata() != 0
    strength = np.abs(values)
    if args.model == 'cross-sectional':
        eligible = pimage.get_fdata() < .005
    else:
        eligible = strength >= 2.807034

    components, _ = label(inside & eligible,
                          structure=generate_binary_structure(3, 1))
    counts = np.bincount(components.ravel())
    clusters = []
    for component in np.flatnonzero(counts[1:] >= extent) + 1:
        voxels = components == component
        peak = np.unravel_index(np.argmax(np.where(voxels, strength, -np.inf)),
                                values.shape)
        clusters.append((int(counts[component]), peak, voxels))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(f'{args.output}_clusters.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['cluster', 'voxels', 'peak_x', 'peak_y', 'peak_z', 'statistic'])
        for i, (count, peak, voxels) in enumerate(sorted(clusters, key=lambda c: -c[0]), 1):
            writer.writerow([i, count, *nib.affines.apply_affine(image.affine, peak), values[peak]])
            nib.save(nib.Nifti1Image(voxels.astype('uint8'), image.affine),
                     f'{args.output}_cluster-{i}.nii.gz')


if __name__ == '__main__':
    main()
