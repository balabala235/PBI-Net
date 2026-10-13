"""Check selected model forward shapes on synthetic inputs; no training."""
import argparse, importlib, sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = ['PBI_Net', 'DLinear', 'PatchTST', 'iTransformer', 'TimesNet', 'TimeMixer', 'FreTS']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', choices=MODELS, default=['PBI_Net'])
    args = p.parse_args()
    config = SimpleNamespace(task_name='long_term_forecast', seq_len=96, label_len=48, pred_len=24, enc_in=1, dec_in=1, c_out=1, d_model=128, e_layers=2, d_ff=256, n_heads=8, factor=3, dropout=0.1, activation='gelu', embed='timeF', freq='h', channel_independence=1, moving_avg=25, top_k=5, num_kernels=6, decomp_method='moving_avg', down_sampling_method='avg', down_sampling_window=2, down_sampling_layers=3, use_norm=1, sap_periods='12,24', sap_bandwidth=1.0, pbi_center_bins='8,4', pbi_mask_mode='physical', pbi_ablation='full', pbi_mlp_hidden=261)
    for name in args.models:
        torch.manual_seed(1001)
        model = importlib.import_module('models.' + name).Model(config).eval()
        x = torch.randn(2, 96, 1)
        x_mark = torch.zeros(2, 96, 4)
        y = torch.zeros(2, 72, 1)
        y_mark = torch.zeros(2, 72, 4)
        with torch.no_grad():
            output = model(x, x_mark, y, y_mark)
        if isinstance(output, tuple):
            output = output[0]
        if tuple(output.shape) != (2, 24, 1) or not torch.isfinite(output).all():
            raise RuntimeError(f'{name}: invalid output')
        print(name, 'PASS shape=', tuple(output.shape), 'parameters=', sum((p.numel() for p in model.parameters())))
        del model
if __name__ == '__main__':
    main()
