"""Run the four deterministic baselines; no neural training."""
import argparse, re, subprocess, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=['djougou', 'medicina', 'membach'], default=['djougou', 'medicina', 'membach'])
    a = p.parse_args()
    rows = []
    for station in a.datasets:
        for method, name in [('naive12', 'SeasonalNaive12'), ('naive24', 'SeasonalNaive24'), ('harmonic1224', 'Harmonic1224')]:
            cmd = [sys.executable, 'periodic_baselines.py', '--data', str(ROOT / 'dataset' / station / f'{station}_gravity_longest.csv'), '--target', 'value', '--method', method, '--seq_len', '96', '--pred_len', '24']
            run = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            print(run.stdout)
            match = re.search('mse:([^,]+),\\s*mae:([^,]+)', run.stdout)
            if not match:
                raise RuntimeError('Metrics missing')
            rows.append(dict(dataset=station.capitalize(), model=name, variant='full', mse=float(match[1]), mae=float(match[2])))
    out = ROOT / 'reproduced_results'
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'simple_baselines.csv', index=False)
    subprocess.run([sys.executable, 'tools/run_multitide_harmonic_baseline.py', '--project', str(ROOT)], cwd=ROOT, check=True)
if __name__ == '__main__':
    main()
