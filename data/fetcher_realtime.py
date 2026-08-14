"""akshare 实时分钟线数据源（盘中实时信号用）。

baostock 分钟线当日更新有延迟（通常接近收盘才可用），无法在开盘 30 分钟
（10:00）出实时信号；这里接入 akshare 的新浪实时分钟接口，盘中即可拿到
当日 10:00 前的真实 bar，配合 IntradayLSTM 输出当日买卖信号。

接口：ak.stock_zh_a_minute（新浪），盘中实时更新，约保留最近 ~41 个交易日，
足够当日 10:00 截面（6 根 bar）+ 前 5 日同时段均量的计算。

字段对齐：返回项目统一 schema（date/time/code/open/high/low/close/volume/amount），
volume 单位为「股」、amount 为「元」，与 baostock 一致，保证 vol_ratio 可比。
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_realtime_min(code: str, frequency: str = "5") -> pd.DataFrame:
    """拉取单只股票的实时分钟线，返回项目统一 schema。

    参数：
        code      : 股票代码，带市场前缀，如 sz.000100
        frequency : 分钟线频率，支持 5/15/30/60

    返回 DataFrame：date/time/code/open/high/low/close/volume/amount，
    按 (date, time) 升序；接口失败或无数据返回空 DataFrame。
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("未安装 akshare，请运行 pip install akshare")
        return pd.DataFrame()

    symbol = code.replace(".", "")  # sz.000100 -> sz000100
    try:
        # 前复权与 baostock adjust="2" 对齐；盘中当日无除权，价格与不复权一致
        raw = ak.stock_zh_a_minute(symbol=symbol, period=frequency, adjust="qfq")
    except Exception as exc:  # noqa: BLE001
        logger.error("akshare 实时分钟拉取 %s 失败: %s", code, exc)
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "time": pd.to_datetime(raw["day"]),
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw["volume"], errors="coerce"),
            "amount": pd.to_numeric(raw.get("amount"), errors="coerce"),
        }
    )
    out["date"] = out["time"].dt.normalize()
    out["code"] = code

    cols = ["date", "time", "code", "open", "high", "low", "close", "volume", "amount"]
    return out[cols].dropna(subset=["close"]).sort_values(["date", "time"]).reset_index(drop=True)
