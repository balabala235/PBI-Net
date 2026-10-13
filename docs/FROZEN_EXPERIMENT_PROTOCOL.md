# Experiment protocol

We use the following settings for the experiments reported in our paper. Run commands from the repository root after preparing the station data and installing the dependencies described in `README.md`.

## Data and forecasting task

- Stations: Djougou, Medicina, and Membach.
- Split: chronological 70% training, 20% validation, and 10% test.
- Input window: 96 h.
- Label context: 48 h.
- Forecast horizon: 24 h.
- Window stride: 1 h, with each window constructed within its split.
- Scaling: `StandardScaler` fitted on the training split only.

We include the processed Djougou and Medicina data. For Membach, follow `dataset/membach/README.md` to obtain and process the official source series. Data attribution and terms are provided in `data/DATA_LICENSE.md`.

## Neural training and evaluation

- Random seeds: 1001–1005.
- Optimizer: Adam.
- Learning rate: `0.001`.
- Batch size: 32.
- Maximum training epochs: 100.
- Early-stopping patience: 10.
- Training loss: MSE.
- Evaluation checkpoint: the checkpoint with the lowest validation loss.

We evaluate the best validation checkpoint on the test split once per seed. The launcher records the commands, logs, prediction arrays, and result-directory entries for each run.

## PBI-Net

We use `d_model=128`, `e_layers=2`, periods `{12, 24}` h, and a Gaussian bandwidth of one FFT bin. The input prior centers are FFT bins `(8, 4)`. The model has 150,712 trainable parameters and predicts the future spectrum without an output mask. The periodic correction head is initialized with zero weights and bias.

Run the full model:

```bash
python tools/run_experiments.py --models PBI_Net --variants full
```

## PBI-Net controls

| Variant | Configuration |
|---|---|
| `no_cfi` | Bypass the sequential affine interaction blocks |
| `period12` | Use the 12 h input prior only |
| `period24` | Use the 24 h input prior only |
| `shift_high` | Set input prior centers to FFT bins `(9, 5)` |
| `shift_low` | Set input prior centers to FFT bins `(7, 3)` |
| `matched_mlp` | Replace interaction with a residual MLP using hidden dimension 261 |
| `neutral_split` | Set every input spectrum allocation weight to `0.5` |
| `far_mask` | Set input prior centers to FFT bins `(16, 12)`, corresponding to 6 h and 8 h |

Run the controls:

```bash
python tools/run_experiments.py --variants no_cfi period12 period24
python tools/run_experiments.py --variants shift_high shift_low matched_mlp
python tools/run_experiments.py --variants neutral_split far_mask
```

We use the same stations, five seeds, and training settings for the full model and controls. The matched MLP has 150,462 trainable parameters. The neutral and far-mask controls retain 150,712 parameters and the one-bin Gaussian bandwidth where applicable.

## Learned baselines

We include DLinear, PatchTST, iTransformer, TimesNet, TimeMixer, and FreTS. Their model-specific settings, calendar inputs, source attribution, and commands are described in `docs/BASELINES.md` and `paper_results/appendix_method_configuration.csv`.

```bash
python tools/run_experiments.py --models DLinear PatchTST iTransformer TimesNet TimeMixer FreTS
```

For CFPT, prepare the compatible external implementation described in `docs/BASELINES.md`, then run:

```bash
python tools/run_experiments.py --models CFPT
```

Add `--dry-run` to a launcher command to display its complete argument lists without starting training.

## Deterministic baselines

SeasonalNaive12 and SeasonalNaive24 repeat the latest observations at periods of 12 h and 24 h, respectively.

Harmonic1224 fits a local regression to each 96 h input window. Its design contains an intercept, a linear trend scaled by `t/95`, and sine–cosine pairs at periods of 12 h and 24 h. We use a fixed ridge penalty of `1e-6`, with the intercept unpenalized. Input time indices are `t=0,...,95`; forecasts are evaluated at `t=96,...,119`.

The eight-constituent harmonic regression uses an intercept, a training-standardized linear trend, and sine–cosine pairs for the following fixed periods:

| Constituent | Period (h) |
|---|---:|
| M2 | 12.4206012 |
| S2 | 12.0000000 |
| N2 | 12.6583475 |
| K2 | 11.9672361 |
| K1 | 23.9344721 |
| O1 | 25.8193387 |
| P1 | 24.0658877 |
| Q1 | 26.8683567 |

We fit coefficients on training data only. The intercept and linear trend are unpenalized. We select the ridge penalty from `{0, 1e-8, 1e-6, 1e-4, 1e-2, 1, 100}` using validation-window MSE, then evaluate the test split. The model extrapolates the training-fitted trend and fixed-frequency phases at future timestamps.

```bash
python tools/run_deterministic.py
```

## Metrics and result summaries

We compute scaled-space MSE and MAE over all test window–horizon pairs. Neural results are reported as the mean and sample standard deviation across five seeds, using `ddof=1`. Deterministic methods are reported as single evaluations.

```bash
python tools/summarize_results.py --manifest reproduced_results/manifest.csv --output reproduced_results/neural_summary.csv
```

We provide our experiment records and reference tables in `paper_results/`. New runs write their summaries to `reproduced_results/`.

## Paired shared-block bootstrap

We compare matched forecast windows and seeds using shared circular time blocks. For each replicate, the same block starts are used across all five paired seeds, followed by resampling the paired seeds with replacement. We use block lengths of 12, 24, and 36 forecast origins, with 10,000 replicates for each comparison.

The difference is PBI-Net error minus comparator error. Negative values favor PBI-Net. We report two-sided 95% percentile intervals. Deterministic harmonic predictions are paired with each PBI-Net seed on the same targets.

After generating the required predictions, run:

```bash
python analysis/shared_block_bootstrap.py --manifest reproduced_results/manifest.csv --comparators neutral_split far_mask matched_mlp shift_high shift_low --output reproduced_results/paired_intervals.csv
```

See `docs/BASELINES.md` for the manifest entries used for learned and deterministic comparators.
