"""盘中预测核心：数据刷新 → 10:00 截面 → IntradayLSTM 推理 → 买卖信号。

从 main.py 的 min-predict 命令抽取，供命令行与调度任务（serve/launchd）复用。
模型为滚动窗口训练（当前 30 分钟 → 预测下一个 30 分钟涨跌），
因此 10:00 信号语义为「未来 30 分钟（10:00→10:30）涨跌」。
非交易日或数据不足时返回 None（调度任务据此优雅跳过）。
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

from config import DATA_DIR, cfg

logger = logging.getLogger(__name__)


def predict_intraday(
    code: str,
    frequency: Optional[str] = None,
    realtime: bool = True,
) -> Optional[dict]:
    """对单只股票做盘中预测，返回信号 dict；无法预测时返回 None。

    参数：
        code      : 股票代码，如 sz.000100
        frequency : 分钟线频率（默认 cfg.min_frequency）
        realtime  : True 用 akshare 实时分钟源（盘中），False 用 baostock

    返回字段：
        code/date/open/price_at_1000/predicted_change_pct/predicted_close/
        prob_close_up/base_pred/action
        （predicted_change_pct/predicted_close/prob_close_up 均为未来 30 分钟语义）
    """
    import numpy as np

    from data.fetcher import fetch_minutes, fetch_stock
    from features.min_builder import build_predict_input

    frequency = frequency or cfg.min_frequency

    # 1. 数据刷新：日线用 baostock（T-1 特征依赖），分钟线默认 akshare 实时源
    end = date.today().isoformat()
    fetch_stock(code, start_date=cfg.start_date, end_date=end, refresh=True)

    realtime_min = None
    if realtime:
        from data.fetcher_realtime import fetch_realtime_min

        realtime_min = fetch_realtime_min(code, frequency)
        if realtime_min.empty:
            logger.warning("akshare 实时分钟为空，回退 baostock 分钟线")
            fetch_minutes(code, start_date=cfg.start_date, end_date=end,
                          frequency=frequency, refresh=True)
            realtime_min = None
        else:
            # 防过期推送：实时源最新交易日不是今天（数据源同步延迟，
            # 如开盘初期当日 bar 未到位）时跳过，避免推送昨日信号
            latest = pd.Timestamp(realtime_min["date"].max()).date()
            if latest != date.today():
                logger.warning(
                    "%s 实时分钟源最新交易日为 %s（非今天），跳过本次预测避免推送过期信号",
                    code, latest,
                )
                return None
            logger.info("已获取 akshare 实时分钟线 %d 根", len(realtime_min))
    else:
        fetch_minutes(code, start_date=cfg.start_date, end_date=end,
                      frequency=frequency, refresh=True)

    # 2. 基座预测（级联特征）
    base_pred = None
    base_path = DATA_DIR / "meta" / "base_preds.parquet"
    if base_path.exists():
        base_preds = pd.read_parquet(base_path)
        hit = base_preds[base_preds["code"] == code].sort_values("date")
        if not hit.empty:
            base_pred = float(hit["base_pred"].iloc[-1])
    if base_pred is None:
        logger.warning("未找到基座预测（%s），base_pred 以 0 填充", code)
        base_pred = 0.0

    # 3. 构造 10:00 截面输入（盘中当日无日线时由内部构造临时日线行）
    inp = build_predict_input(
        code, frequency, base_pred=base_pred, realtime_min=realtime_min
    )
    if inp is None:
        logger.info("%s 无信号（非交易日或数据不足），跳过", code)
        return None

    # 4. 加载分钟模型并推理
    import torch

    from models.nn import IntradayLSTM

    model_path = DATA_DIR / "models" / f"intraday_lstm_{frequency}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"分钟模型不存在: {model_path}，请先运行 min-train")
    ckpt = torch.load(model_path, map_location="cpu")

    net = IntradayLSTM(
        seq_input_dim=ckpt["seq_input_dim"],
        static_dim=len(ckpt["static_cols"]),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    static_vals = np.asarray(
        [inp["static"].get(c, 0.0) for c in ckpt["static_cols"]],
        dtype=np.float32,
    ).reshape(1, -1)
    x_seq = torch.tensor(inp["seq"], dtype=torch.float32)
    x_static = torch.tensor(static_vals, dtype=torch.float32)
    with torch.no_grad():
        pred_reg, pred_cls_logit = net(x_seq, x_static)
    pred_rest = float(pred_reg.squeeze(0).item())
    prob_up = float(torch.sigmoid(pred_cls_logit).squeeze(0).item())
    # 未来 30 分钟预测价（10:00 价 × (1 + 预测涨跌幅)）
    pred_close = inp["close10"] * (1.0 + pred_rest / 100.0)

    # 买卖动作：与网格门控阈值一致（prob_up >= 阈值 允许买入）
    action = "偏多(允许买入)" if prob_up >= cfg.bt_gate_threshold else "偏空(只卖不买)"

    return {
        "code": code,
        "date": str(inp["date"].date()),
        "open": round(inp["open_day"], 2),
        "price_at_1000": round(inp["close10"], 2),
        "predicted_change_pct": round(pred_rest, 3),
        "predicted_close": round(pred_close, 2),
        "prob_close_up": round(prob_up, 3),
        "base_pred": round(base_pred, 4),
        "action": action,
    }
