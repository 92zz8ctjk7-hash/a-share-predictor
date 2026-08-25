"""次日预测（隔夜推送）：用 horizon=1 次日模型预测「明天」涨跌并推送。

与盘中链路（分钟模型预测未来 30 分钟）互补：
- 次日模型（lstm_nextday.pt）：日线特征（今日收盘后可得）→ 预测明日涨跌
- 推送时机：次日早盘前（开盘前拿到当日预判）

数据只依赖日线（收盘后即确定），无实时源延迟问题；
非交易日/数据不足时返回 None（调度任务据此跳过）。
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config import DATA_DIR, cfg

logger = logging.getLogger(__name__)

NEXTDAY_MODEL = "lstm_nextday"


def predict_nextday(code: str) -> Optional[dict]:
    """用次日模型预测下一交易日涨跌，返回信号 dict；无法预测返回 None。

    复用 _predict_with_pretrained（加载 lstm_nextday.pt + 共享标准化），
    取最新交易日（=最近收盘日）的预测作为「明日」预判。

    返回字段：
        code/date/open/predicted_change_pct/prob_up/action
        （date 为预测所基于的最近收盘日；predicted_change_pct 为次日涨跌幅%）
    """
    from backtest.run import _predict_with_pretrained
    from data import store

    model_path = DATA_DIR / "models" / f"{NEXTDAY_MODEL}.pt"
    if not model_path.exists():
        logger.warning("次日模型不存在: %s，请先运行 incremental-update", model_path)
        return None

    bars = store.load_bars(code)
    if bars is None or bars.empty:
        logger.warning("%s 无日线数据", code)
        return None

    sig = _predict_with_pretrained(bars, NEXTDAY_MODEL)
    if sig.empty:
        return None
    last = sig.iloc[-1]
    base_date = pd.Timestamp(last["date"])
    pred_chg = float(last["pred_reg"])
    prob_up = float(last["prob_up"])

    # 动作判定：与门控阈值一致
    action = "偏多(可买入)" if prob_up >= cfg.bt_gate_threshold else "偏空(谨慎买入)"

    return {
        "code": code,
        "date": str(base_date.date()),       # 预测所基于的最近收盘日
        "close": round(float(bars["close"].iloc[-1]), 2),
        "predicted_change_pct": round(pred_chg, 3),
        "prob_up": round(prob_up, 3),
        "action": action,
    }


def format_nextday_message(sig: dict) -> str:
    """次日预测消息（通俗化）。"""
    bullish = "偏多" in sig.get("action", "")
    color = "info" if bullish else "warning"
    chg = sig["predicted_change_pct"]
    direction = "涨" if chg >= 0 else "跌"
    prob_pct = round(sig["prob_up"] * 100)
    pred_close = round(sig["close"] * (1 + chg / 100), 2)
    return (
        f"**{sig['code']} 次日预判**\n"
        f"> 基准收盘（{sig['date']}）: {sig['close']} 元\n"
        f"> 明日预判: 大概率{direction}，到 {pred_close} 元左右（{chg:+.2f}%）\n"
        f"> 上涨把握: {prob_pct}%\n"
        f"> 建议: <font color=\"{color}\">{sig['action']}</font>"
    )


def push_nextday(sig: dict) -> bool:
    """推送次日预测（复用 pushplus 渠道）。"""
    from serving.push import _pushplus_token, push_pushplus, push_wecom

    text = format_nextday_message(sig)
    title = f"次日预判 {sig['code']}"
    channel = cfg.push_channel
    if channel in ("auto", "pushplus") and _pushplus_token():
        return push_pushplus(title, text)
    if channel in ("auto", "wecom"):
        return push_wecom(text)
    return push_wecom(text)
