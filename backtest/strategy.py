"""交易策略：网格交易（可结合模型信号门控）。

网格规则：
- 以基准价 p0 与 range_pct 确定上下界 [lower, upper]，等分 grid_n 格
- 价格越低，目标持仓格数越多（price 位于 upper 时 0 格、位于 lower 时满格）
- 突破上界：目标清仓（止盈）；突破下界：冻结新买入（保留持仓）
- 模型门控（gate_on）：预测上涨概率 < 阈值时只卖不买
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class Strategy(ABC):
    """策略基类：后续可扩展 TopK 组合等策略。"""

    grid_n: int = 1

    @abstractmethod
    def setup(self, base_price: float, capital: float) -> None:
        """回测开始前以基准价与初始资金初始化策略状态。"""

    @abstractmethod
    def target_lots(
        self,
        close: float,
        signal: Optional[Dict[str, float]],
        cur_lots: int,
    ) -> int:
        """依据 T 日收盘价与信号给出目标持仓格数（T+1 开盘执行）。"""


class GridStrategy(Strategy):
    """等分网格策略。"""

    def __init__(
        self,
        grid_n: int = 10,
        range_pct: float = 0.20,
        gate_on: bool = True,
        gate_threshold: float = 0.5,
    ):
        self.grid_n = grid_n
        self.range_pct = range_pct
        self.gate_on = gate_on
        self.gate_threshold = gate_threshold
        self.base_price = 0.0
        self.lower = 0.0
        self.upper = 0.0
        self.step = 0.0

    def setup(self, base_price: float, capital: float) -> None:
        self.base_price = base_price
        self.lower = base_price * (1.0 - self.range_pct)
        self.upper = base_price * (1.0 + self.range_pct)
        self.step = (self.upper - self.lower) / max(self.grid_n, 1)

    def grid_pos(self, price: float) -> int:
        """价格下方的格数（0 ~ grid_n）：价格越低持仓越多。"""
        if price >= self.upper:
            return 0
        if price <= self.lower:
            return self.grid_n
        pos = int((self.upper - price) / self.step)
        return max(0, min(pos, self.grid_n))

    def target_lots(
        self,
        close: float,
        signal: Optional[Dict[str, float]],
        cur_lots: int,
    ) -> int:
        # 突破上界：全部止盈
        if close > self.upper:
            return 0

        pos = self.grid_pos(close)

        # 突破下界：冻结新买入，保留现有持仓
        if close < self.lower:
            pos = min(pos, cur_lots)

        # 模型门控：预测下跌（prob_up < 阈值）时只卖不买
        if self.gate_on and signal is not None:
            if signal.get("prob_up", 0.5) < self.gate_threshold:
                pos = min(pos, cur_lots)

        return pos
