# Baselines

## Six included learned baselines

We include the DLinear, PatchTST, iTransformer, TimesNet, TimeMixer, and FreTS implementations and shared layers used in our experiments. Use the bundled files to run these comparisons. We list the upstream repositories below to acknowledge the original methods and implementations.

Run the six learned baselines and summarize their results:

```bash
python tools/run_experiments.py --models DLinear PatchTST iTransformer TimesNet TimeMixer FreTS
python tools/summarize_results.py --manifest reproduced_results/manifest.csv --output reproduced_results/neural_summary.csv
```

We use the following common experiment settings:

- Three stations: Djougou, Medicina, and Membach.
- Random seeds: 1001–1005.
- Input length: 96 h; label context: 48 h; prediction horizon: 24 h.
- Standardization: `StandardScaler` fitted on the training split only.
- Training loss: MSE.
- Optimizer: Adam with learning rate `0.001`.
- Batch size: 32.
- Maximum training epochs: 100.
- Early-stopping patience: 10.
- Evaluation checkpoint: the checkpoint with the lowest validation loss.

The baseline launcher explicitly sets `d_model=128`, `e_layers=2`, `d_ff=256`, `n_heads=8`, `factor=3`, `dropout=0.1`, `moving_avg=25`, and `channel_independence=1`, where applicable.

Model-specific settings are:

- **TimesNet:** `top_k=5` and `num_kernels=6`.
- **TimeMixer:** average downsampling with `down_sampling_window=2`, `down_sampling_layers=3`, and `use_norm=1`.
- **PatchTST:** constructor defaults of patch length 16 and stride 8.
- **FreTS:** fixed embedding dimension 128 and hidden dimension 256.

See `paper_results/appendix_method_configuration.csv` for the model configurations reported in our paper.

TimesNet, iTransformer, and TimeMixer use normalized hour-of-day, day-of-week, day-of-month, and day-of-year marks from the input window. DLinear, PatchTST, and FreTS use gravity values without calendar marks.

## CFPT: external setup

We provide the CFPT configuration and launcher. The CFPT source is obtained separately from its [official repository](https://github.com/BUPT-SN/CFPT).

In our experiments, we adapted CFPT to the common data and forecasting interface. To use CFPT with this repository, place an authorized compatible implementation at `models/CFPT.py`. It should expose:

- `Model(configs)` as the model constructor.
- `forward(x_enc, x_mark_enc, x_dec, x_mark_dec)` as the forecasting interface.
- Forecast output with shape `[batch, 24, 1]`.

The timestamp branch uses hour-of-day and day-of-week marks for the 24 h forecast horizon, corresponding to the last 24 positions of `x_mark_dec`. Use the time-feature conversion and branch dimensions specified by the experiment implementation.

Our CFPT settings are `d_model=128`, `e_layers=2`, `rda=4`, `rdb=1`, `period=24`, `ksize=5`, and `beta=0.4`. This configuration has 151,361 trainable parameters.

After preparing the compatible local implementation, run:

```bash
python tools/run_experiments.py --models CFPT
```

The launcher checks that `models/CFPT.py` exists and applies the common training settings and CFPT arguments listed above. The public repository provides the configuration and launcher; the adapted CFPT source is obtained separately. When using the official implementation, adapt its constructor, forecasting interface, and timestamp inputs to the interface described here.

## Attribution

We acknowledge the following original methods and implementations:

- **DLinear:** https://github.com/cure-lab/LTSF-Linear
- **PatchTST:** https://github.com/yuqinie98/PatchTST
- **iTransformer:** https://github.com/thuml/iTransformer
- **TimesNet and library implementations:** https://github.com/thuml/Time-Series-Library
- **TimeMixer:** https://github.com/kwuking/TimeMixer
- **FreTS:** https://github.com/aikunyi/FreTS
- **CFPT:** https://github.com/BUPT-SN/CFPT

We preserve applicable source notices and license terms. Our experiment implementations use the Time-Series-Library interfaces. The Apache-2.0 license text is included for components carrying that license, and the root `LICENSE` and `NOTICE` files preserve the framework attribution.

Please cite the relevant baseline papers when using their methods.

## Bootstrap inputs

Neural runs automatically register the following fields in `reproduced_results/manifest.csv`:

```text
dataset,model,variant,seed,result_directory
```

Each result directory contains `pred.npy` and `true.npy` in the same forecast-window order and dataset-standardized space. CFPT runs use the same registration mechanism after the local implementation has been prepared.

For the deterministic eight-constituent harmonic comparator, append one manifest row per station with:

- `dataset`: `Djougou`, `Medicina`, or `Membach`.
- `model`: `MultiTideHarmonicRidge`.
- `variant`: `full`.
- `seed`: leave empty.
- `result_directory`: use the corresponding path recorded in `reproduced_results/multitide_harmonic_results.csv`.

Result-directory paths may be relative to the repository root or absolute. Generate the predictions for PBI-Net and the requested comparators before running the paired bootstrap analysis.
