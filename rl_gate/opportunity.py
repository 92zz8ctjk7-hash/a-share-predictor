"""网格买入机会样本构造：市场状态特征 + 未来 5 日收益标签。

机会定义（纯价格条件，不模拟真实持仓）：
    open[T+1] < close[T] × (1 - step/2)
即次日开盘跌过半个网格间距，可能触发网格买入。

标签：label = (close[T+5] / open[T+1] - 1) × 100（次日开盘买入、5 日后收盘卖出）
正例：label > COST_PCT（覆盖双边交易成本后仍盈利）

特征为 T 日收盘可得的市场状态（与训练/推理口径一致，全部前复权安全）。
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 网格参数（与回测默认一致）：step = 2 × range_pct / grid_n = 4%
GRID_N = 10
RANGE_PCT = 0.20
STEP = 2 * RANGE_PCT / GRID_N

# 双边成本近似（佣金 + 印花税 + 滑点），标签正例阈值
COST_PCT = 0.15

# 持有周期（标签：次日开盘买入 → HORIZON 日后收盘卖出）
HORIZON = 5

# 基础状态特征列（市场状态，训练/推理口径一致）
BASE_FEATURES: List[str] = [
    "ret_1d", "ma_bias_5", "ma_bias_20", "volatility", "volume_ratio",
    "rsi_14", "price_position", "dist_low_20", "amplitude", "turn",
]
# 兼容旧引用：默认全集（含市场环境特征）
FEATURES: List[str] = BASE_FEATURES


def get_features(with_market: bool = True) -> List[str]:
    """特征列：基础 10 维 + （可选）市场环境 12 维（指数环境+日历）。"""
    if not with_market:
        return list(BASE_FEATURES)
    from features.market import MARKET_FEATURES

    return list(BASE_FEATURES) + list(MARKET_FEATURES)


def _add_features(df: pd.DataFrame, with_market: bool = True) -> pd.DataFrame:
    """计算每日市场状态特征（就地修改 df 并返回）。"""
    close = df["close"]

    df["ret_1d"] = close.pct_change()
    df["ma_bias_5"] = close / close.rolling(5).mean() - 1
    df["ma_bias_20"] = close / close.rolling(20).mean() - 1
    df["volatility"] = df["ret_1d"].rolling(20).std()
    df["volume_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    roll_high = df["high"].rolling(20).max()
    roll_low = df["low"].rolling(20).min()
    df["price_position"] = (close - roll_low) / (roll_high - roll_low + 1e-9)
    df["dist_low_20"] = close / roll_low - 1
    df["amplitude"] = (df["high"] - df["low"]) / (
        df["preclose"].replace(0, np.nan) + 1e-9
    )
    # 前复权模式下 baostock 不返回换手率，与历史训练口径一致补 0
    if "turn" not in df.columns or df["turn"].isna().all():
        df["turn"] = 0.0

    # 市场环境特征（指数环境+日历效应，T 日收盘可得）
    if with_market:
        from features.market import MARKET_FEATURES, load_market_features

        market = load_market_features()
        if market is not None:
            df = df.merge(market.reset_index(), on="date", how="left")
            df[MARKET_FEATURES] = df[MARKET_FEATURES].fillna(0.0)
    return df


def _opportunity_one(code: str, bars: pd.DataFrame, with_market: bool = True) -> pd.DataFrame:
    """单只股票的机会样本（全量日期，外层按 cutoff 过滤）。"""
    df = bars.sort_values("date").reset_index(drop=True).copy()
    df = _add_features(df, with_market=with_market)
    close = df["close"]

    # ---- 机会条件与标签 ----
    df["next_open"] = df["open"].shift(-1)
    df["fut_close"] = close.shift(-HORIZON)
    df["is_opportunity"] = df["next_open"] < close * (1 - STEP / 2)
    df["label"] = (df["fut_close"] / df["next_open"] - 1) * 100

    feat_cols = get_features(with_market)
    out = df[df["is_opportunity"] & df["label"].notna()][
        ["date", "code"] + feat_cols + ["label"]
    ].copy()
    out["y"] = (out["label"] > COST_PCT).astype(int)
    return out.dropna(subset=feat_cols).reset_index(drop=True)


def build_day_features(bars: pd.DataFrame, with_market: bool = True) -> pd.DataFrame:
    """每日市场状态特征（gate 推理用，全日期、不做机会过滤）。"""
    df = bars.sort_values("date").reset_index(drop=True).copy()
    df = _add_features(df, with_market=with_market)
    feat_cols = get_features(with_market)
    return df[["date"] + feat_cols].dropna().reset_index(drop=True)


def build_opportunities(
    codes: List[str], before_date=None, with_market: bool = True
) -> pd.DataFrame:
    """对沪深 300 股票联合构造机会样本；before_date 截断防泄漏。"""
    from data import store

    frames = []
    for code in codes:
        bars = store.load_bars(code)
        if bars is None or bars.empty:
            continue
        opp = _opportunity_one(code, bars, with_market=with_market)
        if before_date is not None:
            opp = opp[opp["date"] < before_date]
        if not opp.empty:
            frames.append(opp)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    logger.info(
        "机会样本：%d 条（%d 只股票，正例比例 %.1f%%）",
        len(out), out["code"].nunique(), out["y"].mean() * 100,
    )
    return out
