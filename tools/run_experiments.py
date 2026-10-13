"""Run PBI-Net and controls with the paper settings; save a results manifest."""
import argparse, csv, json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
STATIONS = ['djougou', 'medicina', 'membach']
VARIANTS = {'full': ('12,24', 'full', '8,4', 'physical'), 'no_cfi': ('12,24', 'no_cfi', '8,4', 'physical'), 'period12': ('12', 'full', '8', 'physical'), 'period24': ('24', 'full', '4', 'physical'), 'shift_high': ('12,24', 'full', '9,5', 'physical'), 'shift_low': ('12,24', 'full', '7,3', 'physical'), 'matched_mlp': ('12,24', 'matched_mlp', '8,4', 'physical'), 'neutral_split': ('12,24', 'full', '8,4', 'neutral'), 'far_mask': ('12,24', 'full', '16,12', 'physical')}
FIELDS = ['dataset', 'model', 'variant', 'seed', 'result_directory']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', choices=['PBI_Net', 'DLinear', 'PatchTST', 'iTransformer', 'TimesNet', 'TimeMixer', 'FreTS', 'CFPT'], default=['PBI_Net'])
    p.add_argument('--datasets', nargs='+', choices=STATIONS, default=STATIONS)
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=['full'])
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(1001, 1006)))
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    if any((m != 'PBI_Net' for m in a.models)) and a.variants != ['full']:
        p.error('Baseline models support only --variants full')
    dest = ROOT / 'reproduced_results'
    manifest = dest / 'manifest.csv'
    records = list(csv.DictReader(manifest.open())) if manifest.exists() else []
    for station in a.datasets:
        for model_name in a.models:
            for variant in a.variants:
                for seed in a.seeds:
                    key = (station.capitalize(), model_name, variant, str(seed))
                    existing = [x for x in records if tuple((x[k] for k in FIELDS[:4])) == key]
                    if existing and (not a.force) and (not a.dry_run):
                        d = ROOT / existing[-1]['result_directory']
                        if (d / 'pred.npy').exists() and (d / 'true.npy').exists():
                            print('Skipping completed run', key)
                            continue
                    periods, ablation, bins, mask = VARIANTS[variant]
                    options = {'task_name': 'long_term_forecast', 'is_training': 1, 'model_id': f'{station}_{model_name}_{variant}_seed{seed}', 'model': model_name, 'data': 'Gravity', 'root_path': str(ROOT / 'dataset' / station), 'data_path': f'{station}_gravity_longest.csv', 'features': 'S', 'target': 'value', 'freq': 'h', 'seq_len': 96, 'label_len': 48, 'pred_len': 24, 'enc_in': 1, 'dec_in': 1, 'c_out': 1, 'd_model': 128, 'e_layers': 2, 'sap_periods': periods, 'sap_bandwidth': 1.0, 'pbi_center_bins': bins, 'pbi_mask_mode': mask, 'pbi_ablation': ablation, 'pbi_mlp_hidden': 261, 'learning_rate': 0.001, 'batch_size': 32, 'num_workers': a.num_workers, 'train_epochs': 100, 'patience': 10, 'itr': 1, 'seed': seed, 'gpu': a.gpu, 'des': 'reproduction'}
                    if model_name != 'PBI_Net':
                        options.update(d_ff=256, n_heads=8, factor=3, dropout=0.1, channel_independence=1, moving_avg=25, top_k=5, num_kernels=6)
                    if model_name == 'TimeMixer':
                        options.update(down_sampling_method='avg', down_sampling_window=2, down_sampling_layers=3, use_norm=1)
                    if model_name == 'CFPT':
                        options.update(rda=4, rdb=1, period=24, ksize=5, beta=0.4)
                    if not a.dry_run and (not (ROOT / 'models' / f'{model_name}.py').is_file()):
                        raise FileNotFoundError(f'Missing models/{model_name}.py. See docs/BASELINES.md for CFPT external setup.')
                    cmd = [sys.executable, '-u', 'run.py']
                    for k, v in options.items():
                        cmd.extend(['--' + k, str(v)])
                    if a.device == 'cpu':
                        cmd.append('--no_use_gpu')
                    if a.dry_run:
                        print(json.dumps(cmd))
                        continue
                    if not (ROOT / 'dataset' / station / options['data_path']).is_file():
                        raise FileNotFoundError(options['root_path'])
                    logdir = ROOT / 'logs' / station / model_name / variant
                    logdir.mkdir(parents=True, exist_ok=True)
                    (logdir / f'seed{seed}.command.json').write_text(json.dumps(cmd, indent=2))
                    setting = None
                    with (logdir / f'seed{seed}.log').open('w') as log:
                        process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        for line in process.stdout:
                            print(line, end='')
                            log.write(line)
                            log.flush()
                            match = re.search('>>>>>>>testing : (.*?)<+', line)
                            if match:
                                setting = match.group(1).strip()
                        status = process.wait()
                    if status:
                        raise SystemExit(status)
                    if setting is None:
                        raise RuntimeError('Test result directory was not reported')
                    relative = 'results/' + setting
                    if not all(((ROOT / relative / name).is_file() for name in ['pred.npy', 'true.npy', 'metrics.npy'])):
                        raise RuntimeError('Missing test outputs')
                    records = [x for x in records if tuple((x[k] for k in FIELDS[:4])) != key]
                    records.append(dict(zip(FIELDS, (*key, relative))))
                    dest.mkdir(exist_ok=True)
                    with manifest.open('w', newline='') as f:
                        w = csv.DictWriter(f, FIELDS)
                        w.writeheader()
                        w.writerows(records)
    if not a.dry_run:
        print('Manifest:', manifest)
if __name__ == '__main__':
    main()
