# periodic_baselines.py

import argparse
import numpy as np
import pandas as pd


def seasonal_naive(history, pred_len, period):
    """history: [seq_len]"""
    season = history[-period:]
    repeat_num = int(np.ceil(pred_len / period))
    return np.tile(season, repeat_num)[:pred_len]


def harmonic_features(t, context_len):
    """12/24小时谐波特征，包含截距和局部线性趋势。"""
    t = np.asarray(t, dtype=np.float64)

    # 缩放趋势项，避免数值过大
    trend = t / max(context_len - 1, 1)

    return np.column_stack([
        np.ones_like(t),
        trend,
        np.sin(2.0 * np.pi * t / 12.0),
        np.cos(2.0 * np.pi * t / 12.0),
        np.sin(2.0 * np.pi * t / 24.0),
        np.cos(2.0 * np.pi * t / 24.0),
    ])


def harmonic_12_24(history, pred_len, ridge_alpha=1e-6):
    """
    使用当前输入窗口拟合局部12/24小时谐波回归，
    然后预测未来 pred_len 个时间步。
    """
    history = np.asarray(history, dtype=np.float64)
    seq_len = len(history)

    t_train = np.arange(seq_len)
    t_future = np.arange(seq_len, seq_len + pred_len)

    x_train = harmonic_features(t_train, seq_len)
    x_future = harmonic_features(t_future, seq_len)

    # Ridge解析求解，提高数值稳定性
    penalty = np.eye(x_train.shape[1])
    penalty[0, 0] = 0.0  # 不惩罚截距

    coef = np.linalg.solve(
        x_train.T @ x_train + ridge_alpha * penalty,
        x_train.T @ history
    )

    return x_future @ coef


def predict(history, method, pred_len):
    if method == "naive12":
        return seasonal_naive(history, pred_len, period=12)

    if method == "naive24":
        return seasonal_naive(history, pred_len, period=24)

    if method == "harmonic1224":
        return harmonic_12_24(history, pred_len)

    raise ValueError(f"Unknown method: {method}")


def calculate_metrics(pred, true):
    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    return mse, mae


def main(args):
    df = pd.read_csv(args.data)

    target = args.target if args.target else df.columns[-1]
    values = df[target].astype(np.float64).to_numpy()

    total_len = len(values)

    # ==========================================================
    # 固定划分：训练集、验证集、测试集 = 7:2:1
    # ==========================================================
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.9)

    train_len = train_end
    val_len = val_end - train_end
    test_len = total_len - val_end

    test_start = val_end

    # 仅使用训练集计算标准化参数
    train_values = values[:train_end]
    mean = train_values.mean()
    std = train_values.std()

    if std < 1e-12:
        raise ValueError("Training standard deviation is too small.")

    scaled = (values - mean) / std

    predictions = []
    ground_truths = []

    # 测试样本的输入窗口和预测标签都必须位于测试集内
    for forecast_start in range(
        test_start + args.seq_len,
        total_len - args.pred_len + 1
    ):
        history_start = forecast_start - args.seq_len

        history = scaled[history_start:forecast_start]
        true = scaled[
            forecast_start:forecast_start + args.pred_len
        ]

        if len(history) != args.seq_len:
            continue

        if len(true) != args.pred_len:
            continue

        pred = predict(
            history=history,
            method=args.method,
            pred_len=args.pred_len
        )

        predictions.append(pred)
        ground_truths.append(true)

    predictions = np.asarray(predictions)
    ground_truths = np.asarray(ground_truths)

    if len(predictions) == 0:
        raise ValueError(
            "No valid test samples. "
            "Check seq_len, pred_len and dataset length."
        )

    scaled_mse, scaled_mae = calculate_metrics(
        predictions,
        ground_truths
    )

    # 恢复到原始物理尺度
    original_pred = predictions * std + mean
    original_true = ground_truths * std + mean

    original_mse, original_mae = calculate_metrics(
        original_pred,
        original_true
    )

    expected_samples = (
        test_len - args.seq_len - args.pred_len + 1
    )

    print(f"Method: {args.method}")
    print(f"Target: {target}")
    print(f"Total rows: {total_len}")
    print(
        f"Split rows: train={train_len}, "
        f"val={val_len}, test={test_len}"
    )
    print("Split ratio: 7:2:1")
    print(f"Expected test samples: {expected_samples}")
    print(f"Samples: {len(predictions)}")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Scaled MSE: {scaled_mse:.12f}")
    print(f"Scaled MAE: {scaled_mae:.12f}")
    print(f"Original MSE: {original_mse:.12f}")
    print(f"Original MAE: {original_mae:.12f}")
    print(
        f"mse:{scaled_mse}, "
        f"mae:{scaled_mae}, "
        f"dtw:Not calculated"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--method", type=str, required=True,
                        choices=[
                            "naive12",
                            "naive24",
                            "harmonic1224"
                        ])
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=24)

    main(parser.parse_args())