# Baselines

## Six included learned baselines

We include the DLinear, PatchTST, iTransformer, TimesNet, TimeMixer and FreTS
implementations and shared layers used in our experiments. Use the bundled
files to reproduce these comparisons. We list upstream repositories below
to acknowledge the original methods and implementations.

```bash
python tools/run_experiments.py --models DLinear PatchTST iTransformer TimesNet TimeMixer FreTS
python tools/summarize_results.py --manifest reproduced_results/manifest.csv --output reproduced_results/neural_summary.csv
```

Defaults: three stations, seeds1001..1005, 96/48/24 windows, training-only
StandardScaler, MSE, Adam lr0.001, batch32, max100 epochs, patience10 and best
validation checkpoint. d_model128/e_layers2; other explicit neural settings:
d_ff256, heads8, factor3, dropout0.1, moving_avg25, channel_independence1.
TimesNet top_k5/num_kernels6. TimeMixer avg downsampling/window2/layers3,
use_norm1. PatchTST constructor defaults16/8; FreTS fixed dimensions128/256.
See paper_results/appendix_method_configuration.csv.

Calendar marks: TimesNet/iTransformer/TimeMixer use input-window normalized
hour/weekday/day-of-month/day-of-year. DLinear/PatchTST/FreTS do not use marks.

## CFPT: external setup

CFPT source is not redistributed. Official source:
https://github.com/BUPT-SN/CFPT

In our experiments, we adapted CFPT to the common data and forecasting
interface. We distribute the configuration and launcher separately from the
CFPT source. To run CFPT through this repository,
provide an authorized compatible local models/CFPT.py with Model(configs) and
forward(x_enc,x_mark_enc,x_dec,x_mark_dec), returning [batch,24,1]. Its timestamp
branch must consume hour/weekday marks for the 24 h horizon (the last24 positions
of x_mark_dec), using the same time-feature conversion and branch dimensions as our experiments.
Our CFPT settings: d_model128,e_layers2,rda4,rdb1,period24,ksize5,beta0.4.
Our CFPT configuration has 151,361 trainable parameters.

```bash
python tools/run_experiments.py --models CFPT
```

The command checks that the local model file exists, then applies the shared
training settings and our CFPT arguments. The CFPT comparison requires the
compatible adapted implementation described above. We do not distribute that
adapted source in this release; the official model alone may require interface
adaptation before it can be used with our launcher.

## Attribution

DLinear: https://github.com/cure-lab/LTSF-Linear
PatchTST: https://github.com/yuqinie98/PatchTST
iTransformer: https://github.com/thuml/iTransformer
TimesNet and library implementations: https://github.com/thuml/Time-Series-Library
TimeMixer: https://github.com/kwuking/TimeMixer
FreTS: https://github.com/aikunyi/FreTS

We acknowledge the original baseline authors and preserve applicable source
notices and license terms. Our experiment implementations use the
Time-Series-Library interfaces. Apache-2.0 text is
included for source components carrying that license; the root MIT license
and NOTICE preserve the framework attribution.

## Bootstrap inputs

Neural runs automatically register dataset,model,variant,seed,result_directory
in reproduced_results/manifest.csv. Each result directory contains pred.npy
and true.npy in identical window order and scaled space. The optional CFPT
run uses the same registration mechanism.

For the deterministic eight-constituent comparator, append one manifest row
per station: model MultiTideHarmonicRidge, variant full, empty seed, and
result_directory from reproduced_results/multitide_harmonic_results.csv.
All paths are repository-relative or absolute. Run the bootstrap after all
requested paired predictions have been generated.
