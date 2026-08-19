"""信号融合门控：把板块反转、基座概率等信号合成「是否放行买入」的决策。

设计原则（吸取历史实验教训）：
- 规则少而可解释，每个成分都有独立实证依据，不做黑盒组合
- 板块反转门控纯价格驱动、零模型依赖 → 无前视，可全历史回测
- 基座概率门控需 walk-forward 重训才严格，此处作为可选增强

实证依据（5 年 1222 交易日）：
- 同伴（京东方A/彩虹股份）当日涨幅 vs TCL 次日收益 IC=-0.07（p=0.015），
  板块大涨次日有反转回调风险 → 拦截买入
- 同伴大跌次日略偏强 → 可选放宽（v1 先不放宽，保持保守单侧）
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 板块同伴（与 serving.intraday_signal.PEER_NAMES 一致）
PEER_CODES = ["sz.000725", "sh.600707"]

# 板块反转门控阈值：同伴当日均涨 >= 该值 → 次日拦截买入
PEER_BLOCK_TH = 2.0


def peer_avg_returns(peer_codes: Optional[list] = None) -> pd.Series:
    """各同伴日收益的均值序列（index=date）。纯日线数据，无前视。"""
    from data import store

    peer_codes = peer_codes or PEER_CODES
    frames = []
    for code in peer_codes:
        d = store.load_bars(code)
        if d is None or d.empty:
            logger.warning("同伴 %s 无日线数据", code)
            continue
        d = d.sort_values("date").set_index("date")
        frames.append((d["close"] / d["close"].shift(1) - 1) * 100)
    if not frames:
        return pd.Series(dtype=float)
    return pd.concat(frames, axis=1).mean(axis=1)


def make_peer_gate(
    peer_codes: Optional[list] = None, block_th: float = PEER_BLOCK_TH
):
    """构造板块反转向量门控：buy_gate(day) -> bool。

    引擎在 T+1 开盘撮合时传入 day=T+1（决策依据为 T 日收盘可得信息），
    因此拦截日 = 同伴大涨日 T 的下一交易日：T 日同伴均涨 >= block_th
    时，T+1 的买入被拦截（反转回调风险）；卖出永不受限。
    """
    peer_ret = peer_avg_returns(peer_codes)
    surge = peer_ret >= block_th
    # 大涨日的下一交易日才拦截（与引擎 day=T+1 语义对齐，无前视）
    blocked_days = set(peer_ret.index[surge.to_numpy()].shift(1, freq="B"))
    # 节假日修正：若下一自然工作日非交易日，顺延到实际交易日
    all_days = pd.DatetimeIndex(sorted(peer_ret.index))
    blocked = set()
    for d in blocked_days:
        nxt = all_days[all_days >= d]
        if len(nxt):
            blocked.add(nxt[0])

    def gate(day) -> bool:
        return pd.Timestamp(day) not in blocked

    return gate


def combined_advice(
    grid_action: str,
    prob_up: Optional[float] = None,
    gate_threshold: float = 0.5,
    peer_avg_ret: Optional[float] = None,
    block_th: float = PEER_BLOCK_TH,
) -> Dict:
    """展示层决策汇总：网格动作 + 模型门控 + 板块反转 → 综合建议。

    与回测 GridStrategy 的门控语义一致：
    - 卖出/止盈类动作不受门控限制（只拦截买入）
    - prob_up < 阈值 且 网格想买入 → 降级为观望（模型偏空拦截）
    - 同伴均涨 >= block_th 且 网格想买入 → 降级为观望（板块反转风险）

    返回 {action, reasons[]}：action 为最终建议，reasons 为理由链。
    """
    wants_buy = "买" in grid_action
    reasons = [f"网格: {grid_action}"]
    action = grid_action

    if wants_buy:
        if prob_up is not None and prob_up < gate_threshold:
            action = "观望"
            reasons.append(f"模型偏空(上涨把握{prob_up * 100:.0f}%)拦截买入")
        if peer_avg_ret is not None and peer_avg_ret >= block_th:
            action = "观望"
            reasons.append(f"板块昨日大涨({peer_avg_ret:+.1f}%)，反转风险拦截买入")

    return {"action": action, "reasons": reasons}
