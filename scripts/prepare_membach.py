
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


UNDET_VALUE = 9999.999
COLUMNS = [
    "date",
    "pressure",
    "grav_resid_raw",
    "grav_resid_corr",
    "grav_corr",
    "theory_tide",
]


def read_tsf(path: Path) -> pd.DataFrame:
    rows: list[list[object]] = []
    in_data = False

    with path.open("r", encoding="ascii", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()

            if not in_data:
                if text == "[DATA]":
                    in_data = True
                continue

            if not text:
                continue

            parts = text.split()
            if len(parts) < 11:
                raise ValueError(
                    f"Invalid data row at line {line_no}: expected at least "
                    f"11 fields, got {len(parts)}"
                )

            year, month, day, hour, minute, second = map(int, parts[:6])
            values = list(map(float, parts[6:11]))
            timestamp = pd.Timestamp(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
            )
            rows.append([timestamp, *values])

    if not in_data:
        raise ValueError("The TSF file does not contain a [DATA] section.")
    if not rows:
        raise ValueError("No data rows were parsed from the TSF file.")

    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame = (
        frame.sort_values("date")
        .drop_duplicates(subset="date", keep="first")
        .set_index("date")
    )

    # Preserve the regular hourly time axis. Any absent timestamp becomes NaN.
    full_index = pd.date_range(
        start=frame.index.min(),
        end=frame.index.max(),
        freq="h",
    )
    frame = frame.reindex(full_index)
    frame.index.name = "date"

    value_columns = COLUMNS[1:]
    for column in value_columns:
        frame.loc[np.isclose(frame[column], UNDET_VALUE), column] = np.nan

    return frame.reset_index()


def longest_valid_segment(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    valid = frame[target].notna().to_numpy()

    starts = np.flatnonzero(valid & np.r_[True, ~valid[:-1]])
    ends = np.flatnonzero(valid & np.r_[~valid[1:], True])

    if len(starts) == 0:
        raise ValueError(f"No valid values found in target column {target!r}.")

    lengths = ends - starts + 1
    best = int(np.argmax(lengths))
    start = int(starts[best])
    end = int(ends[best])

    segment = frame.iloc[start : end + 1][["date", target]].copy()
    segment = segment.rename(columns={target: "value"})

    if segment["value"].isna().any():
        raise RuntimeError("Unexpected NaN inside the selected segment.")

    diffs = segment["date"].diff().dropna()
    if not (diffs == pd.Timedelta(hours=1)).all():
        raise RuntimeError("Selected segment is not strictly hourly.")

    return segment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_tsf(args.input)

    all_channels_path = args.output_dir / "membach_all_channels.csv"
    frame.to_csv(
        all_channels_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    segment = longest_valid_segment(frame, target="grav_corr")
    model_path = args.output_dir / "membach_gravity_longest.csv"
    segment.to_csv(
        model_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    print(f"Parsed rows: {len(frame)}")
    print(
        "Full range: "
        f"{frame['date'].iloc[0]} -> {frame['date'].iloc[-1]}"
    )
    print("Missing values:")
    print(frame[COLUMNS[1:]].isna().sum().to_string())
    print()
    print(f"Longest complete gravity segment: {len(segment)} rows")
    print(
        f"Segment range: {segment['date'].iloc[0]} -> "
        f"{segment['date'].iloc[-1]}"
    )
    print(f"Saved all channels: {all_channels_path}")
    print(f"Saved model CSV:   {model_path}")


if __name__ == "__main__":
    main()
