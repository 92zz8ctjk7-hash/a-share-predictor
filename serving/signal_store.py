"""信号持久化：盘中预测信号追加存储到 cache/signals/predictions.parquet。"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)

SIGNAL_DIR = DATA_DIR / "signals"
SIGNAL_FILE = SIGNAL_DIR / "predictions.parquet"

# 去重键：同一生成时刻 + 股票 + 交易日 只保留一条
_DEDUP_KEYS = ["gen_time", "code", "date"]


def save_signal(signal: dict) -> None:
    """追加保存一条预测信号（自动附加生成时间 gen_time）。"""
    row = dict(signal)
    row["gen_time"] = pd.Timestamp.now().floor("s")
    new = pd.DataFrame([row])

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    if SIGNAL_FILE.exists():
        old = pd.read_parquet(SIGNAL_FILE)
        out = (
            pd.concat([old, new], ignore_index=True)
            .drop_duplicates(subset=_DEDUP_KEYS, keep="last")
            .sort_values(["gen_time", "code"])
            .reset_index(drop=True)
        )
    else:
        out = new
    out.to_parquet(SIGNAL_FILE, index=False)
    logger.info("信号已持久化到 %s（累计 %d 条）", SIGNAL_FILE, len(out))


def load_signals(code: Optional[str] = None) -> pd.DataFrame:
    """读取历史信号；不存在返回空 DataFrame。"""
    if not SIGNAL_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(SIGNAL_FILE)
    if code is not None:
        df = df[df["code"] == code]
    return df.reset_index(drop=True)
