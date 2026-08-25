"""生产策略决策引擎：深跌加倍 + logistic 门控 + 延迟卖出（walk 验证 +83%）。

每日 serve 时调用 decide()，输出当日操作建议（展示层，不自动下单）：
- 基座预测：最新 checkpoint 对今日特征推理（未来 5 日涨跌）
- 门控状态：logistic gate（每日增量训练，cache/models/gate_logistic.joblib）
- 网格位置：当前价在 ±20% / 10 格区间中的位置

决策规则（与回测 SellAwareGridStrategy + AggressiveDipStrategy 语义一致）：
    卖出优先：价格在上半区且非强看涨 → 网格卖出（反弹止盈）
    强看涨（pred > +2%）：延迟卖出，持仓观望
    门控拦截 / 浅跌区：攒弹药观望
    深跌区（格数 > 2）且门控放行：加倍买入
    浅跌区且门控放行：正常买入
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from config import DATA_DIR, cfg

logger = logging.getLogger(__name__)

# 与回测一致的参数
GRID_N = cfg.bt_grid_n          # 10 格
RANGE_PCT = cfg.bt_range_pct    # ±20%
SKIP_GRIDS = 2                  # 深跌加倍：跳过浅跌格数
DOUBLE_MULT = 2                 # 深跌区每格筹码倍数
DEFER_TH = 0.02                 # 延迟卖出阈值（未来 5 日预测 > +2%）

# 板块反转调节阈值（同伴昨日均涨≥2% 买入收紧一档 / ≤-2% 放宽一档）
# 实证：京东方 IC-0.070/彩虹-0.053/TCL中环-0.090（三股隔日反转均显著）
PEER_TIGHTEN_TH = 2.0
PEER_LOOSEN_TH = -2.0


def _grid_level(price: float, base_price: float) -> Dict:
    """网格位置：格数（0=上界, GRID_N=下界）、区域、上下界、格线间距。"""
    lower = base_price * (1 - RANGE_PCT)
    upper = base_price * (1 + RANGE_PCT)
    step = (upper - lower) / GRID_N
    if price >= upper:
        pos = 0
    elif price <= lower:
        pos = GRID_N
    else:
        pos = int((upper - price) / step)
        pos = max(0, min(pos, GRID_N))
    if pos == 0:
        zone = "上界(止盈区)"
    elif pos >= GRID_N:
        zone = "下界(冻结区)"
    elif pos <= GRID_N // 2:
        zone = "上半区"
    else:
        zone = "下半区"
    return {"pos": pos, "zone": zone, "upper": round(upper, 2),
            "lower": round(lower, 2), "base": round(base_price, 2),
            "step": round(step, 3)}


def _today_base_pred(code: str) -> Optional[float]:
    """用最新基座 checkpoint 对今日特征推理，返回未来 5 日预测（%）。"""
    try:
        from backtest.run import _predict_with_pretrained
        from data import store

        bars = store.load_bars(code)
        if bars is None or bars.empty:
            return None
        sig = _predict_with_pretrained(bars, "lstm")
        if sig.empty:
            return None
        return float(sig["pred_reg"].iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("基座今日推理失败：%s", exc)
        return None


def _gate_allow(code: str) -> Optional[bool]:
    """logistic gate 对今日的放行判定；模型不存在返回 None（降级）。"""
    try:
        from data import store
        from rl_gate.gate import load_gate, make_buy_gate
        from rl_gate.opportunity import build_day_features, get_features

        gate_bundle = load_gate()
        if gate_bundle is None:
            return None
        bars = store.load_bars(code)
        feats = build_day_features(bars, with_market=True)
        gate_fn = make_buy_gate(gate_bundle, feats.tail(1))
        return bool(gate_fn(feats["date"].iloc[-1]))
    except Exception as exc:  # noqa: BLE001
        logger.warning("gate 今日判定失败：%s", exc)
        return None


def decide(code: str, price: float, anchor: Optional[float] = None,
           n_lots_held: int = 0, peer_avg_prev: Optional[float] = None) -> Dict:
    """生成当日策略决策，返回含 action/依据/波段计划的 dict。

    price ：当前价（盘中实时价或最新收盘价）
    anchor：网格锚定价（影子账户建仓日锁定）；None 时用最新收盘价
    n_lots_held：当前持仓格数（无持仓时卖出类动作降级为观望）
    peer_avg_prev：板块同伴昨日均涨跌(%)，≥+2% 买入收紧一档、≤-2% 放宽一档
    """
    from data import store

    # 网格基准：账户锚定价优先，否则最新收盘（首次建仓场景）
    if anchor is None:
        bars = store.load_bars(code)
        anchor = float(bars["close"].iloc[-1])
    base_price = anchor

    grid = _grid_level(price, base_price)
    pred = _today_base_pred(code)
    gate_ok = _gate_allow(code)

    pos = grid["pos"]
    step = grid["step"]
    action, reason = "观望", "无明确信号"
    buy_lots = 0

    if price >= grid["upper"]:
        action, reason = "清仓止盈", "价格突破网格上界"
    elif price <= grid["lower"]:
        action, reason = "冻结观望", "价格跌破网格下界，暂停买入防套牢"
    elif pos <= GRID_N // 2:
        # 上半区：卖出决策
        if pred is not None and pred > DEFER_TH * 100 and n_lots_held > 0:
            action = "持仓观望(延迟卖出)"
            reason = f"模型强看涨(未来5日 {pred:+.1f}%)，让利润奔跑"
        elif n_lots_held > 0:
            action = "网格卖出"
            reason = f"价格反弹至第 {pos} 格(上半区)，卖出 1 格止盈"
        else:
            action, reason = "观望", "上半区且无持仓，等待回落买入机会"
    else:
        # 下半区：买入决策（浅跌区 = 距下界 SKIP_GRIDS 格以内，与回测 pos<=skip 语义一致）
        if gate_ok is False:
            action, reason = "攒弹药观望", "logistic 门控拦截买入"
        elif pos >= GRID_N - SKIP_GRIDS:
            action, reason = "攒弹药观望", f"近下界浅跌区(第 {pos} 格)，留弹药等深跌"
        elif pos > GRID_N // 2 + SKIP_GRIDS:
            action = "深跌加倍买入"
            buy_lots = DOUBLE_MULT
            reason = f"深跌区(第 {pos} 格)，加倍低吸 {DOUBLE_MULT} 格"
        else:
            action = "正常买入"
            buy_lots = 1
            reason = f"第 {pos} 格，网格正常低吸 1 格"

        # 板块反转调节（隔日反转效应：同伴昨日大涨则今日偏弱）
        if peer_avg_prev is not None and action in ("深跌加倍买入", "正常买入"):
            if peer_avg_prev >= PEER_TIGHTEN_TH:
                if action == "深跌加倍买入":
                    action, buy_lots = "正常买入", 1
                else:
                    action, buy_lots = "观望", 0
                reason += f"；板块昨涨{peer_avg_prev:+.1f}%，收紧一档"
            elif peer_avg_prev <= PEER_LOOSEN_TH:
                if action == "正常买入":
                    action, buy_lots = "深跌加倍买入", DOUBLE_MULT
                reason += f"；板块昨跌{peer_avg_prev:+.1f}%，放宽一档"

    # 波段计划：上一格线（反弹卖出位）/ 下一格线（回落买入位）
    upper_line = round(grid["upper"] - (pos - 1) * step, 2) if pos > 0 else None
    lower_line = round(grid["upper"] - (pos + 1) * step, 2) if pos < GRID_N else None

    return {
        "code": code,
        "price": round(price, 2),
        "grid_pos": pos,
        "grid_zone": grid["zone"],
        "grid_range": f"{grid['lower']}~{grid['upper']}",
        "grid_anchor": round(base_price, 2),
        "base_pred": round(pred, 3) if pred is not None else None,
        "gate_allow": gate_ok,
        "action": action,
        "reason": reason,
        "buy_lots": buy_lots,
        "sell_trigger": upper_line if pos > 0 else None,
        "buy_trigger": lower_line if pos < GRID_N else None,
    }


def run_shadow_flow(
    code: str,
    current_price: float,
    today_open: Optional[float] = None,
    today: Optional[str] = None,
    account_name: str = "shadow",
    peer_avg_prev: Optional[float] = None,
) -> Dict:
    """账户每日执行流：先执行昨日决策（今日开盘价），再生成今日决策。

    与回测同撮合语义：T 日决策 → T+1 开盘成交。首次建仓时锁定网格锚定价，
    清仓止盈后重置锚定。account_name：shadow=影子账户 / real=真实账户。
    peer_avg_prev：板块同伴昨日均涨跌(%)，用于买入档位收紧/放宽。
    返回扩展 advice（含账户快照、分日收益与执行结果）。
    """
    from serving.shadow_account import ShadowAccount

    acc = ShadowAccount.load(account_name)
    exec_txt = None

    # 1. 执行昨日待执行决策（今日开盘价）
    if acc.pending and today and acc.pending.get("date") != today and today_open:
        p = acc.pending
        act = p.get("action", "")
        if "买入" in act and p.get("buy_lots", 0) > 0:
            n = acc.buy(p["buy_lots"], today_open, today)
            if n > 0:
                if acc.grid_anchor is None:  # 首次建仓锁定锚定
                    acc.grid_anchor = today_open
                    acc.anchor_date = today
                lot = acc.lot_shares or 0
                exec_txt = f"开盘执行: 买入 {n} 格({n * lot}股) @ {today_open:.2f}"
        elif act == "网格卖出":
            n = acc.sell(1, today_open, today)
            if n > 0:
                lot = acc.lot_shares or 0
                exec_txt = f"开盘执行: 卖出 {n} 格(约{n * lot}股) @ {today_open:.2f}"
        elif act == "清仓止盈":
            n = acc.sell(1, today_open, today, sell_all=True)
            if n > 0:
                exec_txt = f"开盘执行: 清仓 @ {today_open:.2f}"
                if not acc.lots:
                    acc.grid_anchor = None  # 清仓后重新锚定
        elif act in ("攒弹药观望", "冻结观望", "持仓观望(延迟卖出)", "观望"):
            exec_txt = f"开盘执行: {act}（无操作）"

    # 2. 生成今日决策（用账户锚定与持仓）
    advice = decide(
        code, current_price,
        anchor=acc.grid_anchor,
        n_lots_held=len(acc.lots),
        peer_avg_prev=peer_avg_prev,
    )
    # 每格具体股数：已锁定用锁定值，否则按账户规模估算（初始资金/10格/现价，取整手）
    if acc.lot_shares is not None:
        advice["lot_shares"] = acc.lot_shares
    else:
        advice["lot_shares"] = max(
            int(acc.init_capital / 10 / current_price / 100), 1
        ) * 100
    # 分日收益：先基于旧 prev_equity 算快照，再更新基准
    advice["account"] = acc.snapshot(current_price)
    advice["exec"] = exec_txt
    eq_now = acc.equity(current_price)
    if acc.prev_equity_date != today:
        acc.prev_equity = eq_now
        acc.prev_equity_date = today

    # 3. 保存待执行决策与账户状态
    acc.pending = {"date": today, **{k: advice.get(k) for k in
                    ("action", "buy_lots", "grid_pos")}}
    acc.last_update = today
    acc.save()
    return advice
