# PBI-Net

We provide the research code accompanying our paper, **PBI-Net: Periodic–Background Interaction with Soft Spectral
Priors for Superconducting-Gravimeter Forecasting**. Forecast the next 24 hourly
gravity values from the latest 96 h input. The model uses input-side 12/24 h
spectral priors, two complementary branches and sequential affine interaction.

## Included experiments

PBI-Net; no-interaction and single-period controls; adjacent/far prior shifts;
neutral split; matched-parameter MLP; SeasonalNaive12/24; Harmonic1224;
eight-constituent harmonic regression; DLinear, PatchTST, iTransformer,
TimesNet, TimeMixer and FreTS. We include the model implementations and shared
layers used in our experiments.
CFPT remains an external-source setup; see [baseline instructions](docs/BASELINES.md).
We provide a common training, validation and test interface for these models.

## 1. Install

We conducted our experiments with Python 3.11 and torch 2.5.1+cu121 on an NVIDIA V100.
Create a virtual environment, install a suitable PyTorch build from
https://pytorch.org/get-started/locally/, then:

```bash
pip install -r requirements.txt
```

The direct dependencies are pinned; environment/pip_freeze_full.txt provides
the complete environment snapshot from our experiments. Python launchers support Windows and Linux;
Bash wrappers require Linux/macOS/WSL/Git Bash. CPU runs use --device cpu.

## 2. Data

We include the processed Djougou and Medicina CSVs at:

- dataset/djougou/djougou_gravity_longest.csv
- dataset/medicina/medicina_gravity_longest.csv

For Membach, download the official GWR-C021 TSF series from:
https://seismology.be/en/gravimetry/observations/online-database
Direct file: https://seismology.be/data/MEM_GWRC021.TSF

After saving the TSF locally, run:

```bash
python scripts/prepare_membach.py --input /path/to/MEM_GWRC021.TSF --output-dir dataset/membach
```

Replace the input path with your local file path. The output is
 dataset/membach/membach_gravity_longest.csv. Complete this step before running
all three stations. For a first run with the included data, select
--datasets djougou or --datasets medicina.

Columns: date,value. We document input products, hourly processing and the
paper's data periods in [data preparation](data/README.md).
Djougou/Medicina data are provided with CC BY 4.0 attribution; Membach is
obtained directly from ROB under its data policy. See
[data sources and terms](data/DATA_LICENSE.md).

## 3. Quick start

Run all commands from the repository root:

```bash
python tools/smoke_test.py
python tools/run_experiments.py --datasets djougou --variants full --seeds 1001 --device cpu --num-workers 0
```

The first command checks the forward interface with synthetic inputs. The
second trains and evaluates PBI-Net using our full experiment settings.

## 4. Reproduce PBI-Net experiments

```bash
python tools/run_experiments.py --variants full
python tools/run_experiments.py --variants no_cfi period12 period24
python tools/run_experiments.py --variants shift_high shift_low matched_mlp
python tools/run_experiments.py --variants neutral_split far_mask
```

Defaults: all three stations, seeds 1001–1005, Adam lr 0.001, batch 32,
max 100 epochs, patience 10, best validation checkpoint. Add --device cpu or
--gpu 0 as needed. Use --dry-run to print commands without installing torch
or starting training. Completed manifest entries are skipped; --force reruns.

Outputs: `checkpoints/`, `results/<run>/pred.npy`, `true.npy`, `metrics.npy`;
`logs/<station>/<model>/<variant>/`; `reproduced_results/manifest.csv`.
Metrics array order: MAE,MSE,RMSE,MAPE,MSPE. We identify each run by station,
model, variant and seed. See
[experiment protocol](docs/FROZEN_EXPERIMENT_PROTOCOL.md).

## 5. Baselines

Check the six baseline forward interfaces after installing dependencies:

```bash
python tools/smoke_test.py --models DLinear PatchTST iTransformer TimesNet TimeMixer FreTS
```

Run the six included learned baselines (all stations and five seeds):

```bash
python tools/run_experiments.py --models DLinear PatchTST iTransformer TimesNet TimeMixer FreTS
```

For a single baseline run, add --datasets djougou --seeds 1001 --device cpu
--num-workers 0. We use the constructor defaults and
model configurations reported in our paper; use --dry-run to see the full commands.
We use PatchTST patch length 16 and stride 8. TimeMixer uses average
downsampling with window 2 and three downsampling layers. The baseline
launcher explicitly sets d_ff=256 and factor=3.

### Deterministic baselines

```bash
python tools/run_deterministic.py
```

Runs three simple methods and the eight-constituent method on all stations.
Outputs go to reproduced_results/ and, for the eight-constituent method,
results/. Harmonic1224 uses per-window ridge 1e-6 (intercept unpenalized).
Eight-constituent coefficients use training data only; alpha is selected on
validation data. No astronomical nodal modulation is applied.

## 6. Summarize your runs

```bash
python tools/summarize_results.py --manifest reproduced_results/manifest.csv --output reproduced_results/neural_summary.csv
```

Means and sample SD (ddof=1) are computed from per-run saved prediction arrays.
Single deterministic evaluations receive no seed SD. To regenerate the main-comparison summary from our per-seed experiment records:

```bash
python tools/summarize_results.py --records paper_results/results_summary.csv --output reproduced_results/reference_main_summary.csv
```

We provide our per-seed records and reference tables in paper_results/.
New executions write their outputs to reproduced_results/. Our paper reports
sample standard deviations (ddof=1), summarized in main_comparison_sample_sd.csv.
The earlier results_mean_std.csv table uses ddof=0 and is included for reference.

## 7. Shared block bootstrap

```bash
python analysis/shared_block_bootstrap.py --manifest reproduced_results/manifest.csv --comparators neutral_split far_mask matched_mlp shift_high shift_low --output reproduced_results/paired_intervals.csv
```

Run full and those controls first. The script uses paired target arrays,
shared circular blocks 12/24/36 and 10000 replicates, followed by paired seed
resampling. PatchTST and TimesNet runs are registered automatically in the common manifest.
For CFPT and eight-constituent comparisons, follow docs/BASELINES.md.
For the eight-constituent row use model MultiTideHarmonicRidge, variant full,
empty seed and its result_directory from reproduced_results/multitide_harmonic_results.csv.
Omit --comparators only after all nine comparator inputs are available.

## Directory guide

models/ and exp/: model/training/evaluation; data_provider/: loading;
scripts/: preprocessing and Bash wrappers; tools/: Python entry points;
analysis/: bootstrap; configs/ and docs/: settings and usage;
paper_results/: our experiment records and reference tables; environment/: environment snapshot.

## Attribution and license

We build our forecasting framework on THUML Time-Series-Library (MIT). We
adapt the sequential affine interaction formulation from CFPT (Kou et al.,
2025). Please cite our paper and the relevant baseline papers when using this
repository.
Retain the upstream copyright/license in LICENSE and the Apache notices for
the time-feature utility; see NOTICE. Observation-data terms are specified separately in data/DATA_LICENSE.md.
