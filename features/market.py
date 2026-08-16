"""零成本因子：指数市场环境 + 日历效应。

指数市场环境：沪深 300 指数（sh.000300）衍生特征，为个股提供大盘语境
（A 股联动性强，指数状态对个股短周期方向有显著上下文价值）。
日历效应：星期 one-hot（以周五为基准）+ 月初/月末标记。

全部特征在 T 日收盘可得（指数与个股同步收盘），
日线样本、分钟滚动样本、gate 状态特征均可注入。
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from config import cfg

logger = logging.getLogger(__name__)

# 沪深 300 指数代码（baostock 格式）
INDEX_CODE = "sh.000300"

# 指数市场环境特征列
INDEX_FEATURES: List[str] = [
    "idx_ret_1d", "idx_ret_5d", "idx_ret_20d",
    "idx_ma_bias_20", "idx_volatility_20", "idx_price_position",
]

# 日历效应特征列（dow_1..4 = 周一至周四 one-hot，周五为基准）
CALENDAR_FEATURES: List[str] = [
    "dow_1", "dow_2", "dow_3", "dow_4", "month_start", "month_end",
]

MARKET_FEATURES: List[str] = INDEX_FEATURES + CALENDAR_FEATURES


def ensure_index_bars() -> Optional[pd.DataFrame]:
    """确保沪深 300 指数日线已缓存，返回指数日线。"""
    from data import store

    bars = store.load_bars(INDEX_CODE)
    if bars is None or bars.empty:
        from datetime import date

        from data.fetcher import fetch_stock

        logger.info("拉取沪深300指数 %s ...", INDEX_CODE)
        bars = fetch_stock(
            INDEX_CODE, cfg.start_date, date.today().isoformat(),
            frequency="d", adjust="3", use_cache=True,
        )
    return bars


def build_index_features() -> Optional[pd.DataFrame]:
    """指数市场环境特征，index=date（T 日收盘可得）。"""
    bars = ensure_index_bars()
    if bars is None or bars.empty:
        logger.warning("沪深300指数数据不可用，市场环境特征跳过")
        return None

    df = bars.sort_values("date").reset_index(drop=True)
    close = df["close"]
    out = pd.DataFrame({"date": df["date"]})
    out["idx_ret_1d"] = close.pct_change()
    out["idx_ret_5d"] = close.pct_change(5)
    out["idx_ret_20d"] = close.pct_change(20)
    out["idx_ma_bias_20"] = close / close.rolling(20).mean() - 1
    out["idx_volatility_20"] = close.pct_change().rolling(20).std()
    roll_high = df["high"].rolling(20).max()
    roll_low = df["low"].rolling(20).min()
    out["idx_price_position"] = (close - roll_low) / (roll_high - roll_low + 1e-9)
    return out.dropna().set_index("date")


def build_calendar_features(dates) -> pd.DataFrame:
    """日历效应特征，index=date。"""
    idx = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=idx)
    dow = idx.dayofweek  # 0=周一
    for d in range(4):  # 周一至周四 one-hot，周五为基准避免共线
        out[f"dow_{d + 1}"] = (dow == d).astype(float)
    out["month_start"] = (idx.day <= 3).astype(float)
    out["month_end"] = (idx.day >= 28).astype(float)
    out.index.name = "date"
    return out


def load_market_features() -> Optional[pd.DataFrame]:
    """合并指数 + 日历特征，index=date；指数不可用时返回 None。"""
    idx_feat = build_index_features()
    if idx_feat is None:
        return None
    cal = build_calendar_features(idx_feat.index)
    return pd.concat([idx_feat, cal], axis=1)
