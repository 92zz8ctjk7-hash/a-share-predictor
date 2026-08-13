"""数据结构定义。

这里用 dataclass 定义 A 股的核心数据结构，同时提供与 pandas
DataFrame / numpy / torch tensor 之间的转换方法，方便特征工程和模型训练。

字段说明（与 baostock 日线字段对齐）：
    date      : 交易日期
    code      : 股票代码（带市场前缀，如 sh.600000）
    open      : 开盘价
    high      : 最高价
    low       : 最低价
    close     : 收盘价
    preclose  : 前收盘价
    volume    : 成交量（股）
    amount    : 成交额（元）
    turn      : 换手率（%）
    pct_chg   : 当日涨跌幅（%）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# baostock 日线输出字段 → 内部统一命名
FIELD_MAP: Dict[str, str] = {
    "date": "date",
    "code": "code",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "preclose": "preclose",
    "volume": "volume",
    "amount": "amount",
    "traderStatistic": "turn",
    "pctChg": "pct_chg",
}


@dataclass
class Bar:
    """单日 K 线数据。"""

    date: str
    code: str
    open: float
    high: float
    low: float
    close: float
    preclose: float
    volume: float
    amount: float
    turn: float
    pct_chg: float

    def to_dict(self) -> Dict:
        return {
            "date": self.date,
            "code": self.code,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "preclose": self.preclose,
            "volume": self.volume,
            "amount": self.amount,
            "turn": self.turn,
            "pct_chg": self.pct_chg,
        }


@dataclass
class Stock:
    """单只股票及其历史行情。"""

    code: str
    name: str = ""
    bars: List[Bar] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """转换为按日期升序排列的 DataFrame。"""
        df = pd.DataFrame([b.to_dict() for b in self.bars])
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        return df


@dataclass
class FeatureSample:
    """一个训练/推理样本。

    features : 特征向量（长度 = 特征数量）
    label_reg: 回归标签（未来 N 日涨跌幅，%）
    label_cls: 分类标签（1 = 上涨，0 = 下跌）
    date     : 样本对应的日期（用于时间顺序切分）
    code     : 股票代码
    """

    features: np.ndarray
    label_reg: float
    label_cls: int
    date: str
    code: str

    def to_array(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.features, dtype=np.float32),
            np.asarray([self.label_reg], dtype=np.float32),
            np.asarray([self.label_cls], dtype=np.int64),
        )


def min_bars_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """将 baostock 分钟线返回的 DataFrame 统一为内部字段，并做类型转换。

    baostock 分钟线字段（与日线不同，无 preclose/turn/pctChg）：
        date, time, code, open, high, low, close, volume, amount, adjustflag
    其中 time 形如 "20230103093500000"（YYYYMMDDHHMMSSmmm）。

    返回处理后的 DataFrame：date/time 为 datetime64[ns]，价量为 float64，
    按 date、time 升序排列。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    if "time" in out.columns:
        # baostock time 为 17 位：YYYYMMDDHHMMSS + 3 位毫秒
        out["time"] = pd.to_datetime(out["time"], format="%Y%m%d%H%M%S%f", errors="coerce")

    ordered = [
        "date", "time", "code", "open", "high", "low", "close",
        "volume", "amount",
    ]
    cols = [c for c in ordered if c in out.columns]
    cols += [c for c in out.columns if c not in ordered]
    return out[cols].sort_values(["date", "time"]).reset_index(drop=True)


def bars_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """将 baostock 返回的 DataFrame 统一为内部字段名，并做类型转换。

    返回处理后的 DataFrame（列名已统一），便于后续直接使用。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.rename(columns=FIELD_MAP).copy()

    numeric_cols = [
        "open", "high", "low", "close", "preclose",
        "volume", "amount", "turn", "pct_chg",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])

    # turn（换手率）仅在不复权(3)时由 baostock 返回；
    # 复权模式下缺失时补 NaN，保持下游列结构统一
    if "turn" not in out.columns:
        out["turn"] = np.nan

    # 保持字段顺序一致
    ordered = [
        "date", "code", "open", "high", "low", "close",
        "preclose", "volume", "amount", "turn", "pct_chg",
    ]
    cols = [c for c in ordered if c in out.columns]
    cols += [c for c in out.columns if c not in ordered]
    return out[cols]