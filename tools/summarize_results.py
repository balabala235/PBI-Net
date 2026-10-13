"""Summarize saved test arrays or our per-seed experiment CSV (sample SD)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_mutually_exclusive_group(required=True)
    s.add_argument('--manifest', type=Path)
    s.add_argument('--records', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.records:
        frame = pd.read_csv(a.records)
    else:
        manifest = pd.read_csv(a.manifest)
        rows = []
        for row in manifest.to_dict('records'):
            d = Path(row['result_directory'])
            d = d if d.is_absolute() else ROOT / d
            pred = np.load(d / 'pred.npy')
            true = np.load(d / 'true.npy')
            if pred.shape != true.shape:
                raise ValueError('Prediction/target shape mismatch')
            error = pred - true
            row.update(mse=float(np.mean(error ** 2)), mae=float(np.mean(np.abs(error))))
            rows.append(row)
        frame = pd.DataFrame(rows)
    keys = ['dataset', 'model', 'variant']
    output = []
    if frame.duplicated(keys + ['seed']).any():
        raise ValueError('Duplicate seed records')
    for key, g in frame.groupby(keys, dropna=False, sort=True):
        n = len(g)
        row = dict(zip(keys, key))
        row.update(n=n, seeds='|'.join((str(int(s)) for s in g.seed.dropna().sort_values())), sd_ddof=1)
        for metric in ['mse', 'mae']:
            row[metric + '_mean'] = float(g[metric].mean())
            row[metric + '_std'] = float(g[metric].std(ddof=1)) if n > 1 else np.nan
        output.append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(a.output, index=False)
    print('Wrote', a.output)
if __name__ == '__main__':
    main()
