# Membach

Download the GWR-C021 gravity series from the Royal Observatory of Belgium:
https://seismology.be/en/gravimetry/observations/online-database
Direct TSF link: https://seismology.be/data/MEM_GWRC021.TSF

Save the file locally, then run from the repository root:

```bash
python scripts/prepare_membach.py --input /path/to/MEM_GWRC021.TSF --output-dir dataset/membach
```

Replace /path/to/MEM_GWRC021.TSF with the downloaded file's local path.
The script selects corrected gravity (Vgrav #corr), restores the hourly axis,
converts missing-value markers to NaN, selects the longest continuous valid
segment and writes membach_gravity_longest.csv with columns date,value.
Our paper uses the source snapshot identified as MEM_GWRC021_raw.TSF and the
segment 2014-11-28 02:00 to 2018-03-27 19:00 (29178 hourly observations).
The source series is maintained by ROB. The snapshot period above identifies
our paper's data selection; use the original source coverage for that selection.

Data terms: https://www.seismology.be/en/legal-notices/data-policy
Our repository provides the preprocessing program and source links. Readers
obtain the Membach observations directly from ROB under its data policy.
