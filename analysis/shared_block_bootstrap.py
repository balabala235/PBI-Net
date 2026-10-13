"""Paired shared circular-block bootstrap from a run manifest."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('Djougou', 'Medicina', 'Membach')
COMPARATORS = ('CFPT', 'PatchTST', 'TimesNet', 'neutral_split', 'far_mask', 'multitide', 'matched_mlp', 'shift_high', 'shift_low')
SEEDS = (1001, 1002, 1003, 1004, 1005)

def window_losses(pred: np.ndarray, true: np.ndarray) -> dict[str, np.ndarray]:
    axes = tuple(range(1, pred.ndim))
    return {'mse': np.mean((pred - true) ** 2, axis=axes, dtype=np.float64), 'mae': np.mean(np.abs(pred - true), axis=axes, dtype=np.float64)}

def boot_shared(differences: np.ndarray, block_length: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Shared circular time blocks followed by paired seed resampling."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f'Expected [n_seeds, n_windows], got {values.shape}')
    n_seeds, n_windows = values.shape
    if not 1 <= block_length <= n_windows:
        raise ValueError((block_length, n_windows))
    full_blocks, remainder = divmod(n_windows, block_length)
    extended = np.concatenate([values, values[:, :block_length - 1]], axis=1)
    csum = np.concatenate([np.zeros((n_seeds, 1), dtype=np.float64), np.cumsum(extended, axis=1)], axis=1)
    block_sums = csum[:, block_length:block_length + n_windows] - csum[:, :n_windows]
    starts = rng.integers(n_windows, size=(reps, full_blocks))
    total = block_sums[:, starts].sum(axis=2) if full_blocks else np.zeros((n_seeds, reps), dtype=np.float64)
    if remainder:
        rem_extended = np.concatenate([values, values[:, :remainder - 1]], axis=1) if remainder > 1 else values
        rem_csum = np.concatenate([np.zeros((n_seeds, 1), dtype=np.float64), np.cumsum(rem_extended, axis=1)], axis=1)
        rem_sums = rem_csum[:, remainder:remainder + n_windows] - rem_csum[:, :n_windows]
        rem_starts = rng.integers(n_windows, size=reps)
        total += rem_sums[:, rem_starts]
    time_means = (total / n_windows).T
    seed_draws = rng.integers(n_seeds, size=(reps, n_seeds))
    return time_means[np.arange(reps)[:, None], seed_draws].mean(axis=1)

def load_array_pair(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    pred = np.load(directory / 'pred.npy').astype(np.float64)
    true = np.load(directory / 'true.npy').astype(np.float64)
    if pred.shape != true.shape:
        raise AssertionError((directory, pred.shape, true.shape))
    return (pred, true)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--comparators', nargs='+', choices=COMPARATORS, default=list(COMPARATORS))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    frame = pd.read_csv(a.manifest)
    output = []

    def losses(dataset, model, variant, seed):
        selected = frame[(frame.dataset == dataset) & (frame.model == model) & (frame.variant == variant)]
        if seed is not None:
            selected = selected[selected.seed == seed]
        if len(selected) != 1:
            raise ValueError(f'Expected one manifest row: {dataset}, {model}, {variant}, {seed}; found {len(selected)}')
        directory = Path(str(selected.iloc[0].result_directory))
        directory = directory if directory.is_absolute() else ROOT / directory
        pred, true = load_array_pair(directory)
        return (window_losses(pred, true), true)
    for di, dataset in enumerate(DATASETS):
        for comparator in a.comparators:
            ci = COMPARATORS.index(comparator)
            differences = {'mse': [], 'mae': []}
            for seed in SEEDS:
                baseline, target = losses(dataset, 'PBI_Net', 'full', seed)
                if comparator == 'multitide':
                    other, other_target = losses(dataset, 'MultiTideHarmonicRidge', 'full', None)
                elif comparator in ('CFPT', 'PatchTST', 'TimesNet'):
                    other, other_target = losses(dataset, comparator, 'full', seed)
                else:
                    other, other_target = losses(dataset, 'PBI_Net', comparator, seed)
                if target.shape != other_target.shape or not np.allclose(target, other_target, atol=5e-07, rtol=0):
                    raise ValueError('Paired targets differ')
                for metric in differences:
                    differences[metric].append(baseline[metric] - other[metric])
            for mi, metric in enumerate(('mse', 'mae')):
                values = np.vstack(differences[metric])
                for block in (12, 24, 36):
                    rng_seed = 202609080 + di * 10000 + ci * 100 + mi * 40 + block
                    samples = boot_shared(values, block, 10000, np.random.default_rng(rng_seed))
                    low, high = np.quantile(samples, [0.025, 0.975])
                    output.append(dict(dataset=dataset, comparator=comparator, metric=metric, block_length=block, n_windows=values.shape[1], n_seeds=5, delta_mean=float(values.mean()), ci_low=float(low), ci_high=float(high), bootstrap_reps=10000, rng_seed=rng_seed))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(a.output, index=False)
    print('Wrote', a.output)
if __name__ == '__main__':
    main()
