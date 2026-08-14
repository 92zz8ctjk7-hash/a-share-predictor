"""因子有效性分析：IC / ICIR / 正占比 / 分层收益。

- 多股（>= 2 只）：逐日截面 Spearman 相关（因子值 vs 未来收益）得 IC 序列
- 单股：滚动窗口时序 Spearman 相关（因子值序列 vs 未来收益序列）
- 分层收益：按因子值 5 分位分组，各组未来收益均值（验证因子单调性）
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import DATA_DIR
from features.builder import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def _ic_series_cross_section(df: pd.DataFrame, factor: str) -> np.ndarray:
    """多股模式：逐日截面 Spearman 相关（每日截面样本不足 3 时记 NaN）。"""
    out = []
    for _, g in df.groupby("date"):
        if len(g) < 3:
            out.append(np.nan)
            continue
        coef = spearmanr(g[factor], g["label_reg"]).correlation
        out.append(float(coef) if np.isfinite(coef) else np.nan)
    return np.asarray(out, dtype=np.float64)


def _ic_series_rolling(df: pd.DataFrame, factor: str, rolling: int) -> np.ndarray:
    """单股模式：滚动窗口时序 Spearman 相关。"""
    df = df.sort_values("date").reset_index(drop=True)
    vals = df[factor].to_numpy(dtype=np.float64)
    labs = df["label_reg"].to_numpy(dtype=np.float64)

    out = []
    for i in range(rolling, len(df) + 1):
        a, b = vals[i - rolling:i], labs[i - rolling:i]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        coef = spearmanr(a, b).correlation
        if np.isfinite(coef):
            out.append(float(coef))
    return np.asarray(out, dtype=np.float64)


def _quantile_returns(df: pd.DataFrame, factor: str, n_q: int = 5) -> List[float]:
    """按因子值分位分组，返回各组未来收益均值（%）。"""
    try:
        q = pd.qcut(df[factor], n_q, labels=False, duplicates="drop")
    except ValueError:
        return [np.nan] * n_q
    return [
        round(float(df.loc[q == i, "label_reg"].mean()), 3)
        if (q == i).any() else np.nan
        for i in range(n_q)
    ]


def factor_ic_report(
    sample: pd.DataFrame,
    rolling: int = 60,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    """计算全部特征因子的 IC 指标并按 |IC 均值| 降序输出。

    参数：
        sample  : 样本表（含 date/code/特征列/label_reg）
        rolling : 单股模式下的滚动窗口长度
        out_csv : 结果保存路径（默认 data/meta/factor_ic.csv）
    """
    factors = [c for c in FEATURE_COLUMNS if c in sample.columns]
    n_codes = sample["code"].nunique()
    # 截面 IC 需要足够的股票数（每日截面至少 5 只才有意义），
    # 否则退化为逐股滚动时序 IC 后合并
    use_cross = n_codes >= 5
    mode = "cross-section" if use_cross else f"rolling-{rolling}"
    logger.info("因子 IC 分析：%d 个因子，%d 只股票，模式=%s", len(factors), n_codes, mode)

    rows = []
    for f in factors:
        sub = sample[["date", "code", f, "label_reg"]].dropna()
        if len(sub) < rolling + 1:
            continue

        if use_cross:
            ic = _ic_series_cross_section(sub, f)
            ic = ic[~np.isnan(ic)]
        else:
            parts = [
                _ic_series_rolling(g, f, rolling)
                for _, g in sub.groupby("code")
            ]
            ic = np.concatenate([p for p in parts if len(p) > 0]) if parts else np.array([])

        if len(ic) == 0:
            continue
        ic_mean = float(np.mean(ic))
        ic_std = float(np.std(ic))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        pos_rate = float(np.mean(ic > 0))

        q_rets = _quantile_returns(sub, f)
        rows.append(
            {
                "factor": f,
                "ic_mean": round(ic_mean, 4),
                "ic_std": round(ic_std, 4),
                "icir": round(icir, 3),
                "ic_pos_rate": round(pos_rate, 3),
                **{f"q{i + 1}_ret_pct": q_rets[i] for i in range(len(q_rets))},
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        logger.warning("无有效因子 IC 结果")
        return report

    report = report.reindex(
        report["ic_mean"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    out_csv = out_csv or str(DATA_DIR / "meta" / "factor_ic.csv")
    report.to_csv(out_csv, index=False)  # noqa: PD011
    logger.info("因子 IC 报告已保存到 %s（模式=%s）", out_csv, mode)
    return report
