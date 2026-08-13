"""评估指标与简单信号回测。

指标分类：
- 回归（幅度预测）：MSE / RMSE / MAE / R²
- 分类（方向预测）：Accuracy / Precision / Recall / F1 / AUC
- 回测（简单策略）：基于预测方向构建多头信号，计算组合收益与基准对比
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()

    n = len(y_true)
    if n == 0:
        return {}

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r2": r2,
    }


def cls_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """y_pred 为 0/1，y_score 为预测为正类的概率（用于 AUC）。"""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

    n = len(y_true)
    if n == 0:
        return {}

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    out = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if y_score is not None:
        try:
            from sklearn.metrics import roc_auc_score

            if len(np.unique(y_true)) >= 2:
                out["auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:  # noqa: BLE001
            out["auc"] = float("nan")

    return out


def backtest(
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray,
    y_pred_dir: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """简单策略回测。

    策略：若预测方向为上涨（y_pred_dir=1）则持有标的多头，否则空仓。
    用预测到的幅度作为权重（等权求和），与买入持有基准对比。

    返回累计收益、基准收益、以及策略的超额收益。
    """
    y_true_reg = np.asarray(y_true_reg, dtype=np.float32).ravel()
    y_pred_reg = np.asarray(y_pred_reg, dtype=np.float32).ravel()

    if y_pred_dir is None:
        y_pred_dir = (y_pred_reg > 0).astype(np.int64)
    else:
        y_pred_dir = np.asarray(y_pred_dir, dtype=np.int64).ravel()

    # 持仓信号下的当日实际涨跌幅（%）
    daily = np.where(y_pred_dir == 1, y_true_reg, 0.0)

    # 等价于全样本平均，作为组合收益的近似
    strat_return = float(np.mean(daily))
    bench_return = float(np.mean(y_true_reg))
    excess = strat_return - bench_return

    # 持仓胜率
    held = daily[daily != 0]
    win_rate = float(np.mean(held > 0)) if len(held) > 0 else 0.0

    return {
        "strategy_mean_return_pct": strat_return,
        "benchmark_mean_return_pct": bench_return,
        "excess_return_pct": excess,
        "win_rate": win_rate,
        "n_signals": int(np.sum(y_pred_dir == 1)),
    }


def evaluate_all(
    y_reg_true: np.ndarray,
    y_reg_pred: np.ndarray,
    y_cls_true: np.ndarray,
    y_cls_pred: np.ndarray,
    y_cls_score: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """一次性返回回归指标、分类指标与回测结果。"""
    return {
        "regression": reg_metrics(y_reg_true, y_reg_pred),
        "classification": cls_metrics(y_cls_true, y_cls_pred, y_cls_score),
        "backtest": backtest(y_reg_true, y_reg_pred, y_cls_pred),
    }