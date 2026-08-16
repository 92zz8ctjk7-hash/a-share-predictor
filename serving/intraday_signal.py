"""P3 盘中实盘信号：拟合日内轨迹 + 强度信号 + 网格建议，供 10:00 推送。

设计（主策略=纯网格，轨迹/强度作为辅助参考）：
- 拟合轨迹：用 30 分钟窗口轨迹解码器预测当日剩余路径（期望路径 + σ 不确定带）；
  窗口历史不足时退化为「基座 base_pred 期望方向 + 已实现波动率带」
- 强度信号：历史已实现波动率 → 波动档位（低波正常 / 高波谨慎）
- 网格建议：当前价相对网格区间的位置 → 买入/卖出/观望建议
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import DATA_DIR, cfg

logger = logging.getLogger(__name__)

# 强度档位（已实现波动率分位）
VOL_LOW_TH = 0.33   # 低于 33 分位 = 低波
VOL_HIGH_TH = 0.67  # 高于 67 分位 = 高波


def _realized_vol(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """已实现波动率序列（过去 window 日收益 std）。"""
    df = bars.sort_values("date")
    return df["close"].pct_change().rolling(window).std()


def _vol_regime(bars: pd.DataFrame, window: int = 20) -> Dict:
    """当前波动档位：用历史波动分位判断当前波动水平。"""
    vol = _realized_vol(bars, window).dropna()
    if vol.empty:
        return {"level": "未知", "current": 0.0, "pctile": 0.5}
    current = float(vol.iloc[-1])
    pctile = float((vol < current).mean())
    if pctile <= VOL_LOW_TH:
        level = "低波动(可正常网格)"
    elif pctile >= VOL_HIGH_TH:
        level = "高波动(建议轻仓谨慎)"
    else:
        level = "中波动(正常网格)"
    return {"level": level, "current": round(current, 5), "pctile": round(pctile, 3)}


def _grid_suggestion(price: float, base_price: float, range_pct: float, grid_n: int) -> Dict:
    """当前价相对网格区间的位置 → 网格建议。"""
    lower = base_price * (1 - range_pct)
    upper = base_price * (1 + range_pct)
    grid_step = (upper - lower) / grid_n
    pos = (upper - price) / grid_step  # 价格下方格数
    if price >= upper:
        action, reason = "卖出/止盈", "价格突破网格上界，建议分批止盈"
    elif price <= lower:
        action, reason = "观望/谨慎", "价格跌破网格下界，暂停买入防套牢"
    else:
        # 价格越低，越倾向买入
        if pos >= grid_n * 0.6:
            action, reason = "买入", "价格处于网格低位，可分批买入"
        elif pos <= grid_n * 0.4:
            action, reason = "卖出", "价格处于网格高位，可分批卖出"
        else:
            action, reason = "观望", "价格处于网格中位，持仓观望"
    return {"action": action, "reason": reason, "grid_pos": round(float(pos), 2)}


def predict_intraday_signal(
    code: str,
    grid_n: Optional[int] = None,
    range_pct: Optional[float] = None,
) -> Optional[Dict]:
    """生成盘中信号：拟合轨迹 + 强度信号 + 网格建议。

    返回 dict（供推送）；数据不足返回 None。
    """
    from data import store
    from data.fetcher_realtime import fetch_realtime_min
    from rl_gate.intensity_rl import build_realized_sigma

    grid_n = grid_n or cfg.bt_grid_n
    range_pct = range_pct or cfg.bt_range_pct

    # 实时分钟线（当日）
    rt = fetch_realtime_min(code, cfg.min_frequency)
    if rt.empty:
        logger.warning("%s 无实时分钟数据", code)
        return None
    today = rt["date"].max()
    today_bars = rt[rt["date"] == today].sort_values("time")
    if today_bars.empty:
        return None

    current_price = float(today_bars["close"].iloc[-1])
    open_price = float(today_bars["open"].iloc[0])
    cur_time = today_bars["time"].iloc[-1]

    # 日线（强度 + 网格基准）
    bars = store.load_bars(code)
    if bars is None or bars.empty:
        logger.warning("%s 无日线数据", code)
        return None

    # 网格基准：以昨日收盘为基准
    base_price = float(bars["close"].iloc[-1])

    # 强度信号（已实现波动率档位）
    vol_info = _vol_regime(bars, window=20)

    # 网格建议
    grid_info = _grid_suggestion(current_price, base_price, range_pct, grid_n)

    # 基座期望方向（base_pred）
    base_pred = 0.0
    bp_path = DATA_DIR / "meta" / "base_preds.parquet"
    if bp_path.exists():
        bp = pd.read_parquet(bp_path)
        hit = bp[bp["code"] == code].sort_values("date")
        if not hit.empty:
            base_pred = float(hit["base_pred"].iloc[-1])
    exp_dir = "偏多" if base_pred > 0 else "偏空"

    # 拟合轨迹：用轨迹解码器预测剩余路径（窗口不足时退化为期望方向+波动带）
    trajectory = _fit_trajectory(code, today_bars, base_pred, vol_info["current"])

    return {
        "code": code,
        "date": str(pd.Timestamp(today).date()),
        "time": str(pd.Timestamp(cur_time).strftime("%H:%M")),
        "current_price": round(current_price, 2),
        "open_price": round(open_price, 2),
        "base_price": round(base_price, 2),
        "base_pred_pct": round(base_pred, 2),
        "base_direction": exp_dir,
        "vol_level": vol_info["level"],
        "vol_current": vol_info["current"],
        "vol_pctile": vol_info["pctile"],
        "grid_action": grid_info["action"],
        "grid_reason": grid_info["reason"],
        "grid_pos": grid_info["grid_pos"],
        "trajectory": trajectory,
    }


def _fit_trajectory(
    code: str, today_bars: pd.DataFrame, base_pred: float, vol: float,
) -> Dict:
    """拟合当日剩余轨迹：期望路径 + σ 不确定带。

    优先用 30 分钟窗口轨迹解码器；窗口历史不足时退化为
    「base_pred 期望方向 + 已实现波动率带」。
    返回 {method, expected_close, path_upper_pct, path_lower_pct}。
    """
    from rl_gate.trajectory import (build_window_returns,
                                     build_window_trajectory_samples,
                                     train_trajectory)

    # 尝试用窗口轨迹解码器（需当日已有 >= lookback+1 个窗口）
    try:
        win_ret = build_window_returns(today_bars)
        n_win_today = win_ret["win_idx"].nunique() if not win_ret.empty else 0
        # 窗口足够多才有意义（>= 4 个窗口）
        if n_win_today >= 4:
            # 用当日已有窗口做短期外推（简化：用当日数据拟合）
            r_seq, cond, target, win_idx, meta = build_window_trajectory_samples(
                win_ret, None, code, lookback=3
            )
            if len(r_seq) > 10:
                model, _ = train_trajectory(r_seq, cond, target, win_idx, epochs=30)
                from rl_gate.trajectory import predict_future
                r_obs = win_ret.sort_values("win_idx")["r"].to_numpy()
                fut = predict_future(model, r_obs, 0.0, day_len=8)
                if not fut.empty:
                    cum_mu = float(np.cumsum(fut["mu"].to_numpy())[-1]) if len(fut) else 0.0
                    cum_sigma = float(np.sqrt(np.sum(fut["sigma"].to_numpy()**2))) if len(fut) else 0.0
                    return {
                        "method": "trajectory_decoder",
                        "expected_rest_pct": round(cum_mu * 100, 2),
                        "band_pct": round(cum_sigma * 100, 2),
                    }
    except Exception as exc:  # noqa: BLE001
        logger.debug("轨迹解码器不可用，退化: %s", exc)

    # 退化：base_pred 期望方向 + 已实现波动率带
    return {
        "method": "base_pred+vol_band",
        "expected_rest_pct": round(base_pred, 2),
        "band_pct": round(vol * 100, 2),
    }


def format_intraday_message(sig: Dict) -> str:
    """将盘中信号格式化为推送消息（拟合轨迹 + 强度信号 + 网格建议）。"""
    traj = sig.get("trajectory", {})
    exp_rest = traj.get("expected_rest_pct", 0.0)
    band = traj.get("band_pct", 0.0)
    method = traj.get("method", "")
    exp_close = sig["current_price"] * (1 + exp_rest / 100)

    # 动作颜色
    act = sig["grid_action"]
    color = "info" if "买" in act else ("warning" if "卖" in act else "comment")

    lines = [
        f"**盘中信号 {sig['code']}** ({sig['date']} {sig['time']})",
        f"> 现价: {sig['current_price']}  开盘: {sig['open_price']}",
        f"> 基座方向: {sig['base_direction']} ({sig['base_pred_pct']:+.2f}%)",
        f"> 拟合轨迹: 剩余预期 {exp_rest:+.2f}% (±{band:.2f}%) → 预期收盘 {exp_close:.2f}",
        f"> 强度信号: {sig['vol_level']} (波动分位 {sig['vol_pctile']:.0%})",
        f"> 网格建议: <font color=\"{color}\">{act}</font>",
        f"> {sig['grid_reason']} (网格位置 {sig['grid_pos']})",
    ]
    return "\n".join(lines)


def push_intraday_signal(sig: Dict) -> bool:
    """推送盘中信号（按配置渠道：pushplus 优先，其次企业微信）。"""
    from config import cfg
    from serving.push import _pushplus_token, push_pushplus, push_wecom

    text = format_intraday_message(sig)
    title = f"盘中信号 {sig['code']}"
    channel = cfg.push_channel
    if channel in ("auto", "pushplus") and _pushplus_token():
        return push_pushplus(title, text)
    if channel in ("auto", "wecom"):
        return push_wecom(text)
    return push_wecom(text)
