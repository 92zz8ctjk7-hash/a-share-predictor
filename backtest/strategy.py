"""交易策略：网格交易（可结合模型信号门控）。

网格规则：
- 以基准价 p0 与 range_pct 确定上下界 [lower, upper]，等分 grid_n 格
- 价格越低，目标持仓格数越多（price 位于 upper 时 0 格、位于 lower 时满格）
- 突破上界：目标清仓（止盈）；突破下界：冻结新买入（保留持仓）
- 模型门控（gate_on）：预测上涨概率 < 阈值时只卖不买
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional


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
        day=None,
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
        buy_gate: Optional[Callable] = None,
    ):
        self.grid_n = grid_n
        self.range_pct = range_pct
        self.gate_on = gate_on
        self.gate_threshold = gate_threshold
        # 外部门控（如 RL gate）：buy_gate(day) 返回 False 时禁止买入，只卖不买
        self.buy_gate = buy_gate
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
        day=None,
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

        # 外部门控（RL gate）：禁止买入时只卖不买
        if self.buy_gate is not None and day is not None:
            if not self.buy_gate(day):
                pos = min(pos, cur_lots)

        return pos


class PatientGridStrategy(GridStrategy):
    """耐心低吸网格：模型看跌时买得更深、更慢，规避持续下跌中的连续吸筹。

    动机（历史低吸分析）：纯网格 27% 的买入在后 5 日继续跌 >5%（接飞刀），
    且下跌环境中连续多日逐格加仓放大回撤。改进：
    - 买得更深：信号 pred_reg（未来 5 日预测，horizon 对齐）< 0 时，
      目标格数下移 depth_shift 格——需要跌得更深才触发买入
    - 买得更持久：看跌期每日最多加 max_add_per_day 格，把弹药摊到更多交易日，
      避免在连续下跌中一次性打满
    - 卖出逻辑与普通网格完全一致（不看模型脸色，反弹照卖）
    """

    def __init__(
        self,
        depth_shift: int = 2,
        max_add_per_day: int = 1,
        **kwargs,
    ):
        # 耐心低吸不依赖 prob_up 门控（用 pred_reg 自主决策）
        kwargs.setdefault("gate_on", False)
        super().__init__(**kwargs)
        self.depth_shift = depth_shift
        self.max_add_per_day = max_add_per_day

    def target_lots(
        self,
        close: float,
        signal: Optional[Dict[str, float]],
        cur_lots: int,
        day=None,
    ) -> int:
        # 突破上界：全部止盈
        if close > self.upper:
            return 0

        pos = self.grid_pos(close)

        # 突破下界：冻结新买入
        if close < self.lower:
            pos = min(pos, cur_lots)

        # 模型看跌（未来 5 日预测为负）：买得更深 + 更慢
        pred = signal.get("pred_reg", 0.0) if signal else 0.0
        if pred < 0:
            pos = max(0, pos - self.depth_shift)
            pos = min(pos, cur_lots + self.max_add_per_day)

        # 外部门控（可与 logistic gate 叠加）
        if self.buy_gate is not None and day is not None:
            if not self.buy_gate(day):
                pos = min(pos, cur_lots)

        return pos


class AggressiveDipStrategy(GridStrategy):
    """深跌加倍网格（进攻型低吸）：预判回撤深度，浅跌攒弹药、深跌加倍买。

    动机：能忍受更大回撤，但希望借助模型预判把筹码集中买在更低位置，
    在深 V 反弹中获取更大利润（与 PatientGridStrategy 的防御取向相反）。

    规则（信号 pred_reg = 未来 5 日预测，horizon 对齐）：
    - 模型看平/看多：普通网格，逐格买卖
    - 模型看跌（pred_reg < 0，预判还有回撤）：
      * 浅跌区（格数 ≤ skip_grids）：冻结新买入，攒弹药
      * 深跌区（格数 > skip_grids）：每格 double_mult 份筹码（加倍低吸）
    - 卖出逻辑与普通网格一致（反弹照卖，不看模型脸色）
    """

    def __init__(
        self,
        skip_grids: int = 2,
        double_mult: int = 2,
        **kwargs,
    ):
        kwargs.setdefault("gate_on", False)
        super().__init__(**kwargs)
        self.skip_grids = skip_grids
        self.double_mult = double_mult

    def target_lots(
        self,
        close: float,
        signal: Optional[Dict[str, float]],
        cur_lots: int,
        day=None,
    ) -> int:
        # 突破上界：全部止盈
        if close > self.upper:
            return 0

        pos = self.grid_pos(close)

        # 突破下界：冻结新买入
        if close < self.lower:
            pos = min(pos, cur_lots)

        pred = signal.get("pred_reg", 0.0) if signal else 0.0
        if pred < 0:
            if pos <= self.skip_grids:
                # 浅跌区：不新增（保留现有持仓）
                pos = min(pos, cur_lots)
            else:
                # 深跌区：每格加倍筹码，上限 grid_n
                pos = min(self.grid_n, (pos - self.skip_grids) * self.double_mult)

        # 外部门控（可选叠加）
        if self.buy_gate is not None and day is not None:
            if not self.buy_gate(day):
                pos = min(pos, cur_lots)

        return pos


class SellAwareGridStrategy(AggressiveDipStrategy):
    """卖出侧模型化网格：在深跌加倍基础上，用基座预测调节卖出节奏。

    非对称容错设计（卖早=机会成本，卖晚=真实损失）：
    - 看跌加速止盈（宽松触发）：pred_reg < accel_th 且价格在网格上半区时，
      目标格数额外减 accel_shift（比机械网格多卖，抢先兑现）
    - 看涨延迟卖出（严格确认）：pred_reg > defer_th（高门槛）时，
      目标格数加 defer_shift 但不超过当前持仓（只减少卖出，不新增买入）
    - 买入侧逻辑与 AggressiveDipStrategy 完全一致
    """

    def __init__(
        self,
        accel_th: float = 0.0,
        accel_shift: int = 1,
        defer_th: float = 0.02,
        defer_shift: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.accel_th = accel_th
        self.accel_shift = accel_shift
        self.defer_th = defer_th
        self.defer_shift = defer_shift

    def target_lots(
        self,
        close: float,
        signal: Optional[Dict[str, float]],
        cur_lots: int,
        day=None,
    ) -> int:
        pos = super().target_lots(close, signal, cur_lots, day=day)

        pred = signal.get("pred_reg", 0.0) if signal else 0.0
        # 上半区判定：价格位于网格上半部分（格数少于一半）才允许卖出调节
        upper_half = self.grid_pos(close) <= self.grid_n // 2

        if close > self.upper:
            return pos  # 突破上界无条件清仓，不调节

        if pred < self.accel_th and upper_half:
            # 看跌加速止盈：多卖 accel_shift 格
            pos = max(0, pos - self.accel_shift)
        elif pred > self.defer_th:
            # 强看涨延迟卖出：少卖 defer_shift 格（不因此新增买入）
            pos = min(cur_lots, pos + self.defer_shift)

        return pos
