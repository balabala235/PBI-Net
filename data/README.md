# Station data and preparation

We use three continuous hourly gravity series with columns date,value.

| Station | Data preparation | Paper segment | Rows |
|---|---|---|---:|
| Djougou | Included at dataset/djougou/djougou_gravity_longest.csv |2012-04-28 12:00 to 2012-07-12 13:00|1802|
| Medicina | Included at dataset/medicina/medicina_gravity_longest.csv |1998-01-29 11:00 to 2000-05-16 12:00|20114|
| Membach | Download from ROB and run scripts/prepare_membach.py |2014-11-28 02:00 to 2018-03-27 19:00|29178|

Djougou/Medicina use EOST IGETS Level 2(code 32) g_nofil records. We require 60
valid on-grid minute observations for each hourly value and select the longest
continuous valid segment. scripts/prepare_djougou.py and prepare_medicina.py
provide processing entry points; run --help for the source input options.

Membach uses ROB corrected gravity(Vgrav #corr). Follow
[the Membach preparation steps](../dataset/membach/README.md) to create its CSV
before starting three-station runs.

We use chronological 70/20/10 splits, training-only StandardScaler and windows
inside each split: input 96 h, label context 48 h, prediction 24 h.
DATASET_CHECKSUMS.csv identifies the processed files used in our paper.
[Data sources, citations and licenses](DATA_LICENSE.md) describe each provider's
terms separately from the software license.
