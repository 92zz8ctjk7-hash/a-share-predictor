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

# 板块同伴（面板板块，仅展示与反转提示，不参与 serving 预测）
PEER_NAMES = {"sz.000725": "京东方A", "sh.600707": "彩虹股份"}

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


def _peer_context() -> Dict:
    """板块同伴动态：今日早盘走势（实时）+ 昨日涨跌反转提示。

    实证依据（5 年 1222 天）：板块同日联动 IC 0.6~0.8，但同伴早盘走势
    对 TCL 剩余时段预测力≈0；唯一显著信号是隔日反转
    （同伴昨日涨幅 vs TCL 今日 IC≈-0.07，p=0.015）：
    同伴昨日大涨 → 今日谨防回调（买入收紧），反之低吸可放宽。
    任一环节取数失败时优雅降级，不影响主信号。
    """
    from data import store
    from data.fetcher_realtime import fetch_realtime_min

    peers: Dict = {}
    prev_rets = []
    for code, name in PEER_NAMES.items():
        info: Dict = {"name": name}
        try:
            d = store.load_bars(code)
            if d is not None and len(d) >= 2:
                prev_ret = (float(d["close"].iloc[-1]) / float(d["close"].iloc[-2]) - 1) * 100
                info["prev_ret"] = round(prev_ret, 2)
                prev_rets.append(prev_ret)
            rt = fetch_realtime_min(code, cfg.min_frequency)
            if not rt.empty:
                today = rt["date"].max()
                tb = rt[rt["date"] == today].sort_values("time")
                if not tb.empty:
                    open_p = float(tb["open"].iloc[0])
                    cur = float(tb["close"].iloc[-1])
                    info["morning_ret"] = round((cur / open_p - 1) * 100, 2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("同伴 %s 数据获取失败: %s", code, exc)
        peers[code] = info

    avg_prev = sum(prev_rets) / len(prev_rets) if prev_rets else 0.0
    if avg_prev >= 2.0:
        hint = f"板块昨日大涨({avg_prev:+.1f}%)，历史有反转效应，今日买入建议收紧一档"
    elif avg_prev <= -2.0:
        hint = f"板块昨日大跌({avg_prev:+.1f}%)，历史有反转效应，今日低位承接可放宽一档"
    else:
        hint = ""
    return {"peers": peers, "avg_prev_ret": round(avg_prev, 2), "reversal_hint": hint}


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
    # 新鲜度守卫：实时源未同步到今天（开盘初期数据源延迟是常态）时跳过，
    # 避免把昨日盘中参考当作今日推送（与 predict_intraday 同款守卫）
    if pd.Timestamp(today).date() != date.today():
        logger.warning(
            "%s 实时分钟源最新交易日为 %s（非今天），跳过盘中参考避免推送过期内容",
            code, pd.Timestamp(today).date(),
        )
        return None
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

    # 综合建议：生产策略决策引擎 + 双账户（影子模拟 + 真实跟踪，walk 验证 +83%）
    try:
        from serving.shadow_account import account_path
        from serving.strategy_advice import run_shadow_flow

        advice = run_shadow_flow(
            code, current_price,
            today_open=float(open_price),
            today=str(pd.Timestamp(today).date()),
            account_name="shadow",
        )
        # 真实账户（存在时才跟踪）
        if account_path("real").exists():
            advice_real = run_shadow_flow(
                code, current_price,
                today_open=float(open_price),
                today=str(pd.Timestamp(today).date()),
                account_name="real",
            )
        else:
            advice_real = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("策略决策生成失败，降级为网格动作：%s", exc)
        advice = {"action": grid_info["action"], "reason": grid_info["reason"],
                  "grid_pos": grid_info["grid_pos"]}
        advice_real = None

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
        "combined_advice": advice,
        "real_account": (advice_real or {}).get("account"),
        "real_exec": (advice_real or {}).get("exec"),
        "real_advice": {k: (advice_real or {}).get(k) for k in
                        ("action", "reason", "buy_lots", "lot_shares")} if advice_real else None,
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
    """将盘中信号格式化为通俗易懂的推送消息（轨迹 + 波动 + 网格建议）。"""
    traj = sig.get("trajectory", {})
    exp_rest = traj.get("expected_rest_pct", 0.0)
    band = traj.get("band_pct", 0.0)
    exp_close = sig["current_price"] * (1 + exp_rest / 100)

    # 开盘以来的涨跌
    open_chg = (sig["current_price"] / sig["open_price"] - 1) * 100 if sig["open_price"] else 0.0
    trend = "上涨" if exp_rest >= 0 else "下跌"

    lines = [
        f"**{sig['code']} 盘中参考**（{sig['date']} {sig['time']}）",
        f"> 现价: {sig['current_price']} 元（开盘 {sig['open_price']} 元，已{open_chg:+.2f}%）",
        f"> 大盘模型看后市: {'看涨' if sig['base_direction'] == '偏多' else '看跌'}"
        f"（预期 {sig['base_pred_pct']:+.2f}%）",
        f"> 今天剩余时间: 预计{trend} {abs(exp_rest):.2f}%，"
        f"收盘大约 {exp_close:.2f} 元（上下浮动 {band:.2f}%）",
        f"> 盘面波动: {sig['vol_level']}（近期波动排在 {sig['vol_pctile']:.0%} 位置）",
    ]
    # 综合建议（生产策略：深跌加倍 + logistic 门控 + 延迟卖出）
    advice = sig.get("combined_advice")
    if advice:
        a_color = "info" if "买" in advice.get("action", "") else (
            "warning" if "卖" in advice.get("action", "") else "comment")
        gate_txt = {
            True: "门控放行", False: "门控拦截", None: "门控未就绪",
        }.get(advice.get("gate_allow"), "门控未就绪")
        pred_txt = (
            f"未来5日 {advice['base_pred']:+.2f}%"
            if advice.get("base_pred") is not None else "基座未就绪"
        )
        lines.append(
            f"> 策略建议: <font color=\"{a_color}\">{advice.get('action', '观望')}</font>"
            f"（{advice.get('reason', '')}）"
        )
        lines.append(
            f"> 决策依据: 网格第 {advice.get('grid_pos', '-')} 格"
            f"({advice.get('grid_range', '')}) | {pred_txt} | {gate_txt}"
        )
        # 影子账户状态（策略模拟验证）
        acct = advice.get("account")
        if acct:
            pos_txt = (
                f"持仓 {acct['n_lots']} 格(成本 {acct['avg_cost']}) 浮盈 {acct['unrealized_pct']:+.1f}%"
                if acct['n_lots'] else "空仓"
            )
            win_txt = f" | 胜率 {acct['win_rate']:.0f}%" if acct.get('win_rate') is not None else ""
            lines.append(
                f"> 影子账户: 总资产 {acct['equity']/10000:.2f}万"
                f"(收益 {acct['total_return_pct']:+.1f}%) | 现金 {acct['cash']/10000:.2f}万 | {pos_txt}"
                f" | 已实现盈亏 {acct['realized_pnl']:+,.0f}元{win_txt}"
            )
        # 真实账户：只推送策略建议与执行动作，不展示资金数额
        racc = sig.get("real_account")
        if racc:
            if sig.get("real_exec"):
                lines.append(f"> 真账执行: {sig['real_exec'].replace('开盘执行: ', '')}")
            radv = sig.get("real_advice")
            if radv:
                r_act = radv.get("action", "观望")
                r_color = "info" if "买" in r_act else (
                    "warning" if "卖" in r_act else "comment")
                r_lot = radv.get("lot_shares")
                r_lot_txt = f"，每格约 {r_lot} 股" if r_lot else ""
                lines.append(f"> 真账建议: <font color=\"{r_color}\">{r_act}</font>"
                             f"（{radv.get('reason', '')}{r_lot_txt}）")
        # 开盘执行结果
        if advice.get("exec"):
            lines.append(f"> {advice['exec']}")
        # 波段计划（具体挂单价位 + 股数）
        swing = []
        lot_shares = advice.get("lot_shares")
        lot_txt = f"(约 {lot_shares} 股)" if lot_shares else ""
        if advice.get("sell_trigger") and acct and acct.get("n_lots"):
            swing.append(f"反弹至 {advice['sell_trigger']} 卖 1 格{lot_txt}")
        if advice.get("buy_trigger"):
            if advice.get("action") == "深跌加倍买入":
                n_buy, hint = 2, "加倍"
            else:
                n_buy, hint = 1, ""
            lots_txt = f"(约 {lot_shares * n_buy} 股)" if lot_shares else ""
            swing.append(f"回落至 {advice['buy_trigger']} 买 {n_buy} 格{hint}{lots_txt}")
        if swing:
            lines.append(f"> 波段计划: {' | '.join(swing)}")
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
