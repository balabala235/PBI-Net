#!/usr/bin/env python3
"""Prepare Medicina EOST Level-2 GGP files for hourly forecasting.

The script reads monthly ``*32.ggp`` files, extracts the real-observation
gravity channel ``g_nofil``, converts the official missing marker 99999.999 to
NaN, calculates strict hourly means from 60 valid one-minute observations, and
extracts the longest gap-free hourly segment.

No interpolation, smoothing, detrending, tide removal, or standardization is
performed here. Standardization should be fitted on the training split only by
the model data loader.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATA_START_MARKER = "77777777"
DATA_BLOCK_END_MARKER = "88888888"
FILE_END_MARKER = "99999999"
MISSING_VALUE = 99999.999
EXPECTED_COLUMNS = 8
EXPECTED_MINUTES_PER_HOUR = 60


@dataclass
class FileStats:
    path: Path
    sha256: str
    has_start_marker: bool
    has_end_marker: bool
    data_rows: int
    malformed_rows: int
    invalid_timestamps: int
    missing_gravity_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Medicina EOST Level-2 monthly GGP files into strict "
            "hourly g_nofil data for time-series forecasting."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the downloaded Level-2 .ggp files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which processed CSV files and the summary are saved.",
    )
    parser.add_argument(
        "--pattern",
        default="IGETS-SG-CORMIN-mc023-*32.ggp",
        help=(
            "Input filename glob. The default selects only EOST product 32 "
            "files and avoids mixing products from different processing centres."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-files",
        action="store_true",
        help=(
            "Allow files without the 99999999 end marker. Do not use this "
            "option for formal experiments."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_timestamp(date_token: str, time_token: str) -> pd.Timestamp | None:
    """Parse GGP yyyymmdd/hhmmss tokens, whose leading zeros may be omitted."""
    if len(date_token) != 8 or not date_token.isdigit():
        return None
    if not time_token.isdigit() or len(time_token) > 6:
        return None

    hhmmss = time_token.zfill(6)
    hour = int(hhmmss[0:2])
    minute = int(hhmmss[2:4])
    second = int(hhmmss[4:6])
    if hour > 23 or minute > 59 or second > 59:
        return None

    try:
        return pd.Timestamp(
            year=int(date_token[0:4]),
            month=int(date_token[4:6]),
            day=int(date_token[6:8]),
            hour=hour,
            minute=minute,
            second=second,
        )
    except ValueError:
        return None


def parse_ggp_file(path: Path) -> tuple[pd.DataFrame, FileStats]:
    timestamps: list[pd.Timestamp] = []
    gravity_values: list[float] = []
    in_data = False
    has_start_marker = False
    has_end_marker = False
    malformed_rows = 0
    invalid_timestamps = 0
    missing_gravity_rows = 0
    data_rows = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()

            # A GGP file contains a data block followed by a file terminator:
            # 77777777 -> observations -> 88888888 -> 99999999.
            if stripped == FILE_END_MARKER:
                has_end_marker = True
                in_data = False
                break

            if stripped == DATA_START_MARKER:
                has_start_marker = True
                in_data = True
                continue

            # End only the observation block here.  Keep reading so that the
            # following 99999999 file-end marker can be verified.
            if stripped == DATA_BLOCK_END_MARKER:
                in_data = False
                continue

            if not in_data or not stripped or stripped.startswith("C"):
                continue

            parts = stripped.split()
            if len(parts) != EXPECTED_COLUMNS:
                malformed_rows += 1
                continue

            data_rows += 1
            timestamp = parse_timestamp(parts[0], parts[1])
            if timestamp is None:
                invalid_timestamps += 1
                continue

            try:
                gravity = float(parts[4])  # g_nofil
            except ValueError:
                malformed_rows += 1
                continue

            if np.isclose(gravity, MISSING_VALUE, rtol=0.0, atol=1e-6):
                gravity = np.nan
                missing_gravity_rows += 1

            timestamps.append(timestamp)
            gravity_values.append(gravity)

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps),
            "value": np.asarray(gravity_values, dtype="float64"),
        }
    )
    stats = FileStats(
        path=path,
        sha256=sha256_file(path),
        has_start_marker=has_start_marker,
        has_end_marker=has_end_marker,
        data_rows=data_rows,
        malformed_rows=malformed_rows,
        invalid_timestamps=invalid_timestamps,
        missing_gravity_rows=missing_gravity_rows,
    )
    return frame, stats


def validate_files(
    stats: Iterable[FileStats], allow_incomplete_files: bool
) -> None:
    stats = list(stats)
    missing_start = [item.path.name for item in stats if not item.has_start_marker]
    missing_end = [item.path.name for item in stats if not item.has_end_marker]
    malformed = [
        (item.path.name, item.malformed_rows, item.invalid_timestamps)
        for item in stats
        if item.malformed_rows or item.invalid_timestamps
    ]

    errors: list[str] = []
    if missing_start:
        errors.append(
            "Files without the 77777777 data-start marker:\n  - "
            + "\n  - ".join(missing_start)
        )
    if missing_end and not allow_incomplete_files:
        errors.append(
            "Files without the 99999999 end marker (possibly incomplete):\n  - "
            + "\n  - ".join(missing_end)
        )
    if malformed:
        detail = "\n".join(
            f"  - {name}: malformed_rows={bad}, invalid_timestamps={bad_time}"
            for name, bad, bad_time in malformed
        )
        errors.append("Files containing unparseable data rows:\n" + detail)

    if errors:
        raise RuntimeError("\n\n".join(errors))

    if missing_end:
        print(
            "WARNING: incomplete files were allowed explicitly; these outputs "
            "must not be used for formal experiments.",
            file=sys.stderr,
        )


def merge_minutes(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    minutes = pd.concat(frames, ignore_index=True)
    if minutes.empty:
        raise RuntimeError("No parseable data rows were found in the selected files.")

    minutes = minutes.sort_values("date", kind="stable").reset_index(drop=True)
    duplicate_mask = minutes.duplicated("date", keep=False)
    duplicate_rows = int(duplicate_mask.sum())

    if duplicate_rows:
        duplicate_groups = minutes.loc[duplicate_mask].groupby("date", sort=False)
        conflicts: list[pd.Timestamp] = []
        for timestamp, group in duplicate_groups:
            values = group["value"]
            non_missing = values.dropna().unique()
            has_nan = bool(values.isna().any())
            if len(non_missing) > 1 or (has_nan and len(non_missing) == 1):
                conflicts.append(timestamp)
                if len(conflicts) >= 10:
                    break

        if conflicts:
            examples = "\n  - ".join(str(item) for item in conflicts)
            raise RuntimeError(
                "Conflicting duplicate timestamps were found. First examples:\n  - "
                + examples
            )

        minutes = minutes.drop_duplicates("date", keep="first").reset_index(drop=True)

    return minutes, duplicate_rows


def make_hourly_data(minutes: pd.DataFrame) -> pd.DataFrame:
    working = minutes.copy()
    working["hour"] = working["date"].dt.floor("h")
    working["minute_of_hour"] = working["date"].dt.minute
    working["on_minute_grid"] = (
        working["date"].dt.second.eq(0)
        & working["date"].dt.microsecond.eq(0)
    )

    grouped = working.groupby("hour", sort=True)
    hourly = pd.DataFrame(
        {
            "observed_minutes": grouped.size(),
            "valid_minutes": grouped["value"].count(),
            "unique_minutes": grouped["minute_of_hour"].nunique(),
            "on_minute_grid": grouped["on_minute_grid"].all(),
            "value": grouped["value"].mean(),
        }
    )

    full_hours = pd.date_range(
        start=hourly.index.min(), end=hourly.index.max(), freq="h"
    )
    hourly = hourly.reindex(full_hours)
    hourly.index.name = "date"

    for column in ("observed_minutes", "valid_minutes", "unique_minutes"):
        hourly[column] = hourly[column].fillna(0).astype("int64")
    hourly["on_minute_grid"] = hourly["on_minute_grid"].fillna(False).astype(bool)

    hourly["complete_hour"] = (
        hourly["observed_minutes"].eq(EXPECTED_MINUTES_PER_HOUR)
        & hourly["valid_minutes"].eq(EXPECTED_MINUTES_PER_HOUR)
        & hourly["unique_minutes"].eq(EXPECTED_MINUTES_PER_HOUR)
        & hourly["on_minute_grid"]
    )
    hourly.loc[~hourly["complete_hour"], "value"] = np.nan

    return hourly.reset_index()[
        [
            "date",
            "value",
            "observed_minutes",
            "valid_minutes",
            "complete_hour",
        ]
    ]


def find_continuous_segments(
    hourly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = hourly.loc[hourly["complete_hour"], ["date", "value"]].copy()
    if valid.empty:
        raise RuntimeError("No hour contains 60 valid one-minute g_nofil values.")

    new_segment = valid["date"].diff().ne(pd.Timedelta(hours=1))
    valid["segment_id"] = new_segment.cumsum().astype("int64")

    segments = (
        valid.groupby("segment_id", sort=True)
        .agg(start=("date", "min"), end=("date", "max"), hours=("date", "size"))
        .reset_index()
        .sort_values(["hours", "start"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    segments["rank_by_length"] = np.arange(1, len(segments) + 1)
    segments = segments[
        ["rank_by_length", "segment_id", "start", "end", "hours"]
    ]

    longest_id = int(segments.iloc[0]["segment_id"])
    longest = valid.loc[valid["segment_id"].eq(longest_id), ["date", "value"]]
    longest = longest.reset_index(drop=True)
    return segments, longest


def write_sha256_manifest(stats: list[FileStats], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in stats:
            handle.write(f"{item.sha256}  {item.path.name}\n")


def write_summary(
    path: Path,
    input_dir: Path,
    pattern: str,
    stats: list[FileStats],
    minutes: pd.DataFrame,
    duplicate_rows: int,
    hourly: pd.DataFrame,
    segments: pd.DataFrame,
    longest: pd.DataFrame,
    allow_incomplete_files: bool,
) -> None:
    total_source_rows = sum(item.data_rows for item in stats)
    total_missing_gravity = sum(item.missing_gravity_rows for item in stats)
    valid_minute_rows = int(minutes["value"].notna().sum())
    complete_hours = int(hourly["complete_hour"].sum())
    incomplete_hours = int((~hourly["complete_hour"]).sum())
    longest_start = longest["date"].iloc[0]
    longest_end = longest["date"].iloc[-1]

    lines = [
        "Medicina Level-2 preprocessing summary",
        "=" * 42,
        f"Input directory: {input_dir.resolve()}",
        f"Filename pattern: {pattern}",
        f"Selected files: {len(stats)}",
        "Selected product: EOST Level 2, product suffix 32.ggp",
        "Target channel: g_nofil (real valid corrected gravity observations)",
        "Gravity unit: nm/s^2",
        "Time basis: UTC",
        f"Incomplete files explicitly allowed: {allow_incomplete_files}",
        "",
        "Minute-level parsing",
        "-" * 42,
        f"Source data rows: {total_source_rows}",
        f"Unique parsed timestamps: {len(minutes)}",
        f"Duplicate source rows: {duplicate_rows}",
        f"Valid g_nofil minute rows: {valid_minute_rows}",
        f"Missing-marker g_nofil rows: {total_missing_gravity}",
        f"Minute time range: {minutes['date'].min()} to {minutes['date'].max()}",
        "",
        "Strict hourly aggregation",
        "-" * 42,
        "Rule: an hourly mean is valid only with all 60 unique, on-grid, valid minutes",
        f"Full hourly timeline rows: {len(hourly)}",
        f"Complete valid hours: {complete_hours}",
        f"Incomplete or missing hours: {incomplete_hours}",
        f"Continuous valid segments: {len(segments)}",
        "",
        "Longest continuous segment",
        "-" * 42,
        f"Start: {longest_start}",
        f"End: {longest_end}",
        f"Rows/hours: {len(longest)}",
        "Columns: date,value",
        "",
        "Operations intentionally NOT performed",
        "-" * 42,
        "No use of g_fil as target",
        "No interpolation or gap filling",
        "No smoothing or outlier filtering",
        "No detrending or tide removal",
        "No whole-dataset standardization",
        "",
        "Formal model input",
        "-" * 42,
        "Use medicina_gravity_longest.csv for the current forecasting experiments.",
        "Fit the scaler using the training split only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(path for path in input_dir.glob(args.pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No files matched {args.pattern!r} under {input_dir}"
        )

    print(f"Selected {len(files)} files. Parsing g_nofil ...")
    frames: list[pd.DataFrame] = []
    file_stats: list[FileStats] = []
    for index, path in enumerate(files, start=1):
        frame, stats = parse_ggp_file(path)
        frames.append(frame)
        file_stats.append(stats)
        print(
            f"[{index:03d}/{len(files):03d}] {path.name}: "
            f"rows={stats.data_rows}, missing_g={stats.missing_gravity_rows}, "
            f"complete={stats.has_end_marker}"
        )

    validate_files(file_stats, args.allow_incomplete_files)
    minutes, duplicate_rows = merge_minutes(frames)
    hourly = make_hourly_data(minutes)
    segments, longest = find_continuous_segments(hourly)

    output_dir.mkdir(parents=True, exist_ok=True)
    minute_valid_path = output_dir / "medicina_minute_valid.csv"
    hourly_path = output_dir / "medicina_hourly_all.csv"
    segments_path = output_dir / "medicina_continuous_segments.csv"
    longest_path = output_dir / "medicina_gravity_longest.csv"
    summary_path = output_dir / "medicina_processing_summary.txt"
    hashes_path = output_dir / "medicina_raw_files_sha256.txt"

    minutes.loc[minutes["value"].notna(), ["date", "value"]].to_csv(
        minute_valid_path, index=False, date_format="%Y-%m-%d %H:%M:%S"
    )
    hourly.to_csv(hourly_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    segments.to_csv(
        segments_path, index=False, date_format="%Y-%m-%d %H:%M:%S"
    )
    longest.to_csv(longest_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    write_sha256_manifest(file_stats, hashes_path)
    write_summary(
        summary_path,
        input_dir,
        args.pattern,
        file_stats,
        minutes,
        duplicate_rows,
        hourly,
        segments,
        longest,
        args.allow_incomplete_files,
    )

    print("\nProcessing completed.")
    print(f"Minute valid data: {minute_valid_path}")
    print(f"Hourly quality table: {hourly_path}")
    print(f"Continuous segments: {segments_path}")
    print(f"Longest model dataset: {longest_path}")
    print(f"Processing summary: {summary_path}")
    print(f"Raw-file hashes: {hashes_path}")
    print(
        f"Longest segment: {longest['date'].iloc[0]} to "
        f"{longest['date'].iloc[-1]} ({len(longest)} hours)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
