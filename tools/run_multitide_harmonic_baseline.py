#!/usr/bin/env python3
"""Run a leakage-safe multi-constituent harmonic baseline on three SG series.

The constituent set is fixed before evaluation.  Coefficients and the global
StandardScaler equivalent are fitted on the training split only.  Ridge alpha
is selected on the validation split, and the test split is evaluated once
using exactly the same 96/24 window construction as Dataset_Gravity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "Djougou": ("djougou", "djougou_gravity_longest.csv"),
    "Membach": ("membach", "membach_gravity_longest.csv"),
    "Medicina": ("medicina", "medicina_gravity_longest.csv"),
}

# Fixed primary diurnal and semidiurnal constituents.  Periods are in hours.
CONSTITUENTS = (
    ("M2", 12.4206012),
    ("S2", 12.0000000),
    ("N2", 12.6583475),
    ("K2", 11.9672361),
    ("K1", 23.9344721),
    ("O1", 25.8193387),
    ("P1", 24.0658877),
    ("Q1", 26.8683567),
)

DEFAULT_ALPHAS = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0)
SEQ_LEN = 96
PRED_LEN = 24
LABEL_LEN = 48


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_alphas(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("ridge alphas must be finite non-negative numbers")
    if len(set(values)) != len(values):
        raise ValueError("duplicate ridge alphas are not allowed")
    return tuple(sorted(values))


def load_series(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["date", "value"]:
        raise ValueError(f"{path}: expected columns date,value")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    values = frame["value"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite values found")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f"{path}: timestamps are not strictly increasing")
    if len(dates) > 1:
        step_ns = np.diff(dates.asi8)
        expected_ns = 3_600 * 1_000_000_000
        if not np.all(step_ns == expected_ns):
            raise ValueError(f"{path}: series is not continuously hourly sampled")
    return dates, values


def make_design(hours: np.ndarray, trend_center: float, trend_scale: float) -> np.ndarray:
    columns = [
        np.ones_like(hours, dtype=np.float64),
        (hours - trend_center) / trend_scale,
    ]
    for _, period in CONSTITUENTS:
        angle = 2.0 * np.pi * hours / period
        columns.extend((np.cos(angle), np.sin(angle)))
    return np.column_stack(columns)


def fit_ridge(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 0.0:
        return np.linalg.lstsq(design, target, rcond=None)[0]
    gram = design.T @ design
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0  # intercept
    penalty[1, 1] = 0.0  # linear trend
    system = gram + alpha * penalty
    rhs = design.T @ target
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(system) @ rhs


def target_windows(values: np.ndarray) -> np.ndarray:
    count = len(values) - SEQ_LEN - PRED_LEN + 1
    if count <= 0:
        raise ValueError(
            f"split of {len(values)} rows cannot form a {SEQ_LEN}/{PRED_LEN} window"
        )
    windows = [
        values[index + SEQ_LEN:index + SEQ_LEN + PRED_LEN]
        for index in range(count)
    ]
    return np.asarray(windows, dtype=np.float64)[..., None]


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    difference = pred.astype(np.float64) - true.astype(np.float64)
    mse = float(np.mean(difference ** 2))
    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(mse))
    valid = np.abs(true) > 1e-8
    if np.any(valid):
        ratio = difference[valid] / true[valid]
        mape = float(np.mean(np.abs(ratio)))
        mspe = float(np.mean(ratio ** 2))
    else:
        mape = float("nan")
        mspe = float("nan")
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe}


def protocol_payload(dataset: str, data_path: Path, alphas: tuple[float, ...]) -> dict:
    return {
        "method": "eight_constituent_training_fitted_harmonic_regression",
        "dataset": dataset,
        "data_path": str(data_path),
        "data_sha256": file_sha256(data_path),
        "split": [0.7, 0.2, 0.1],
        "seq_len": SEQ_LEN,
        "label_len": LABEL_LEN,
        "pred_len": PRED_LEN,
        "scaler_fit": "training_split_only_ddof0",
        "coefficient_fit": "training_split_only",
        "alpha_selection": "validation_window_mse_only",
        "test_usage": "single_final_evaluation",
        "ridge_alphas": list(alphas),
        "trend": "intercept_plus_unpenalized_linear",
        "constituents": [
            {"name": name, "period_hours": period}
            for name, period in CONSTITUENTS
        ],
    }


def protocol_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_dataset(root: Path, dataset: str, alphas: tuple[float, ...]):
    directory, filename = DATASETS[dataset]
    data_path = root / "dataset" / directory / filename
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    dates, raw = load_series(data_path)
    total = len(raw)
    train_end = int(total * 0.7)
    validation_end = int(total * 0.9)
    split_bounds = {
        "train": [0, train_end],
        "validation": [train_end, validation_end],
        "test": [validation_end, total],
    }

    train_raw = raw[:train_end]
    train_mean = float(train_raw.mean())
    train_std = float(train_raw.std(ddof=0))
    if not train_std > 0:
        raise ValueError(f"{dataset}: training standard deviation is not positive")
    scaled = (raw - train_mean) / train_std

    hours = (dates.asi8 - dates.asi8[0]) / (3_600 * 1_000_000_000)
    hours = hours.astype(np.float64)
    trend_center = float(hours[:train_end].mean())
    trend_scale = float(hours[:train_end].std(ddof=0))
    if not trend_scale > 0:
        raise ValueError(f"{dataset}: trend scale is not positive")

    design = make_design(hours, trend_center, trend_scale)
    x_train = design[:train_end]
    y_train = scaled[:train_end]
    validation_true = target_windows(scaled[train_end:validation_end])

    search_rows = []
    candidates = []
    for alpha in alphas:
        coefficients = fit_ridge(x_train, y_train, alpha)
        validation_prediction_series = design[train_end:validation_end] @ coefficients
        validation_pred = target_windows(validation_prediction_series)
        metrics = regression_metrics(validation_pred, validation_true)
        row = {
            "dataset": dataset,
            "alpha": alpha,
            "validation_mse": metrics["mse"],
            "validation_mae": metrics["mae"],
        }
        search_rows.append(row)
        candidates.append((metrics["mse"], alpha, coefficients))

    _, selected_alpha, coefficients = min(candidates, key=lambda item: (item[0], item[1]))
    test_prediction_series = design[validation_end:] @ coefficients
    test_pred = target_windows(test_prediction_series)
    test_true = target_windows(scaled[validation_end:])
    metrics = regression_metrics(test_pred, test_true)

    setting = f"multitide_harmonic_{dataset.lower()}_96_24"
    result_dir = root / "results" / setting
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload(dataset, data_path, alphas)
    payload.update(
        {
            "protocol_sha256": protocol_sha256(payload),
            "selected_alpha": selected_alpha,
            "rows": total,
            "split_bounds": split_bounds,
            "train_mean_raw": train_mean,
            "train_std_raw": train_std,
            "design_condition_number_unregularized": float(np.linalg.cond(x_train)),
            "setting": setting,
        }
    )

    config_path = result_dir / "config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("protocol_sha256") != payload["protocol_sha256"]:
            raise RuntimeError(
                f"{config_path} has a different protocol; refusing to overwrite"
            )

    np.save(result_dir / "pred.npy", test_pred.astype(np.float32))
    np.save(result_dir / "true.npy", test_true.astype(np.float32))
    np.save(
        result_dir / "metrics.npy",
        np.asarray(
            [metrics["mae"], metrics["mse"], metrics["rmse"], metrics["mape"], metrics["mspe"]],
            dtype=np.float64,
        ),
    )
    np.save(result_dir / "coefficients.npy", coefficients.astype(np.float64))
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    recomputed = regression_metrics(
        np.load(result_dir / "pred.npy"),
        np.load(result_dir / "true.npy"),
    )
    if abs(recomputed["mse"] - metrics["mse"]) > 5e-7:
        raise AssertionError(f"{dataset}: saved prediction MSE mismatch")
    if abs(recomputed["mae"] - metrics["mae"]) > 5e-7:
        raise AssertionError(f"{dataset}: saved prediction MAE mismatch")

    result_row = {
        "dataset": dataset,
        "model": "MultiTideHarmonicRidge",
        "variant": "eight_constituent",
        "deterministic": True,
        "n_test_windows": int(test_pred.shape[0]),
        "pred_shape": "x".join(str(value) for value in test_pred.shape),
        "selected_alpha": selected_alpha,
        "mse_scaled": metrics["mse"],
        "mae_scaled": metrics["mae"],
        "rmse_scaled": metrics["rmse"],
        "mse_raw": metrics["mse"] * train_std ** 2,
        "mae_raw": metrics["mae"] * train_std,
        "rmse_raw": metrics["rmse"] * train_std,
        "train_std_raw": train_std,
        "setting": setting,
        "result_directory": str(result_dir.relative_to(root)),
        "data_sha256": payload["data_sha256"],
        "protocol_sha256": payload["protocol_sha256"],
        "status": "MATCH",
    }
    return result_row, search_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument(
        "--alphas",
        default=",".join(str(value) for value in DEFAULT_ALPHAS),
        help="fixed comma-separated ridge grid selected using validation only",
    )
    args = parser.parse_args()
    root = args.project.resolve()
    alphas = parse_alphas(args.alphas)

    output_dir = root / "reproduced_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    alpha_rows = []
    errors = []
    for dataset in DATASETS:
        try:
            row, search = run_dataset(root, dataset, alphas)
            rows.append(row)
            alpha_rows.extend(search)
            print(
                f"[PASS] {dataset}: mse={row['mse_scaled']:.9f}, "
                f"mae={row['mae_scaled']:.9f}, alpha={row['selected_alpha']}"
            )
        except Exception as exc:
            errors.append({"dataset": dataset, "error": str(exc)})
            print(f"[ERROR] {dataset}: {exc}", file=sys.stderr)

    result_path = output_dir / "multitide_harmonic_results.csv"
    alpha_path = output_dir / "multitide_harmonic_alpha_search.csv"
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(result_path, index=False)
    pd.DataFrame(alpha_rows).to_csv(alpha_path, index=False)

    comparison_path = output_dir / "multitide_vs_pbi_full_comparison.csv"
    comparison_written = False
    authoritative_path = root / "paper_results" / "main_comparison_sample_sd.csv"
    if authoritative_path.is_file() and len(result_frame) == 3:
        authoritative = pd.read_csv(authoritative_path)
        pbi = authoritative[
            (authoritative["model"] == "PBI_Net")
            & (authoritative["variant"] == "full")
        ][["dataset", "mse_mean", "mse_std", "mae_mean", "mae_std"]].copy()
        pbi = pbi.rename(
            columns={
                "mse_mean": "pbi_mse_mean",
                "mse_std": "pbi_mse_sd",
                "mae_mean": "pbi_mae_mean",
                "mae_std": "pbi_mae_sd",
            }
        )
        comparison = result_frame.merge(
            pbi,
            on="dataset",
            how="left",
            validate="one_to_one",
        )
        comparison["harmonic_minus_pbi_mse"] = (
            comparison["mse_scaled"] - comparison["pbi_mse_mean"]
        )
        comparison["harmonic_minus_pbi_mae"] = (
            comparison["mae_scaled"] - comparison["pbi_mae_mean"]
        )
        comparison["harmonic_relative_mse_vs_pbi_pct"] = (
            100.0
            * comparison["harmonic_minus_pbi_mse"]
            / comparison["pbi_mse_mean"]
        )
        comparison["pbi_lower_mse"] = (
            comparison["pbi_mse_mean"] < comparison["mse_scaled"]
        )
        comparison["pbi_lower_mae"] = (
            comparison["pbi_mae_mean"] < comparison["mae_scaled"]
        )
        comparison.to_csv(comparison_path, index=False)
        comparison_written = True

    report = {
        "status": "PASS" if len(rows) == 3 and not errors else "INCOMPLETE",
        "expected_datasets": 3,
        "verified_datasets": len(rows),
        "errors": errors,
        "leakage_controls": {
            "scaler_fit": "training_only",
            "coefficient_fit": "training_only",
            "ridge_selection": "validation_only",
            "constituents": "fixed_before_test",
            "test": "not_used_for_selection",
        },
        "outputs": [
            str(result_path.relative_to(root)),
            str(alpha_path.relative_to(root)),
        ] + ([str(comparison_path.relative_to(root))] if comparison_written else []),
    }
    report_path = output_dir / "multitide_harmonic_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
