"""分层资金计划：盈利放大 + 双回撤刹车（相对/绝对）+ 资金分层处理。

动机（方案 A 的进化）：
- 单纯动态每格（任何时候按总资产折算）验证无效——牛市段每格股数缩水
- 改进：复利只在盈利时生效（E > 初始资金才按总资产折算，亏损期缩仓不放大）
- 双回撤风控：
  * 相对回撤 = (高水位 - 当前) / 高水位，触发后降最大持仓格数
  * 绝对回撤 = (初始资金 - 当前) / 初始资金，深度触发冻结新买入
- 迟滞恢复：触发阈值与恢复阈值分离，防止状态在临界点反复抖动

引擎集成：run_engine(capital_plan=...) 每次撮合前 plan.update(equity)，
eff_lot = plan.lot_size(...)，target 上限 = plan.max_grids(...)，
危险态只卖不买。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TieredCapitalPlan:
    """分层资金计划（状态随账户净值演化，跨日连续）。"""

    init_capital: float = 100000.0
    # 相对回撤（vs 高水位）触发/恢复
    rel_dd_th: float = 0.08
    rel_dd_restore: float = 0.04
    # 绝对回撤（vs 初始资金）触发/恢复
    abs_dd_th: float = 0.10
    abs_dd_restore: float = 0.05
    # 绝对回撤危险线（冻结新买入）/恢复
    danger_abs_th: float = 0.15
    danger_abs_restore: float = 0.10
    # 防御态最大持仓格数比例
    defend_grid_ratio: float = 0.6

    def __post_init__(self):
        self.peak = self.init_capital
        self.state = "normal"  # normal / defend / danger
        self.history = []  # (equity, state) 轨迹，供分析

    # ---- 状态机 ----

    def update(self, equity: float) -> None:
        self.peak = max(self.peak, equity)
        rel_dd = (self.peak - equity) / self.peak if self.peak > 0 else 0.0
        abs_dd = (self.init_capital - equity) / self.init_capital

        if self.state == "normal":
            if abs_dd >= self.danger_abs_th:
                self.state = "danger"
            elif rel_dd >= self.rel_dd_th or abs_dd >= self.abs_dd_th:
                self.state = "defend"
        elif self.state == "defend":
            if abs_dd >= self.danger_abs_th:
                self.state = "danger"
            elif rel_dd <= self.rel_dd_restore and abs_dd <= self.abs_dd_restore:
                self.state = "normal"
        elif self.state == "danger":
            if abs_dd <= self.danger_abs_restore:
                self.state = "defend"

        self.history.append((equity, self.state))

    # ---- 输出 ----

    def allow_buy(self) -> bool:
        """危险态冻结新买入。"""
        return self.state != "danger"

    def max_grids(self, grid_n: int, equity: float = None, price: float = None) -> int:
        """当前状态允许的最大持仓格数。"""
        if self.state == "defend":
            return max(1, int(grid_n * self.defend_grid_ratio))
        return grid_n

    def lot_size(self, equity: float, price: float, grid_n: int,
                 lot_unit: int = 100) -> int:
        """每格股数：盈利区按总资产折算（复利），亏损区封顶初始资金。"""
        base = equity if equity > self.init_capital else min(equity, self.init_capital)
        return max(int(base / max(grid_n, 1) / price / lot_unit), 1) * lot_unit

    def summary(self) -> dict:
        states = [s for _, s in self.history]
        return {
            "final_state": self.state,
            "defend_days": states.count("defend"),
            "danger_days": states.count("danger"),
            "peak": round(self.peak, 2),
        }


@dataclass
class BudgetedCapitalPlan:
    """回撤预算仓位控制（事前预算式，替代事后刹车）。

    核心公式（每次撮合前实时计算）：
        亏损预算   = 当前总资产 - 底线资产
        可承受持仓 = 亏损预算 / 最坏跌幅假设
        最大格数   = 可承受持仓 / 每格金额

    特性：
    - 无悬崖式触发：越接近底线，可买格数平滑收缩至 0（自然冻结）
    - 盈利自动放大：赚得越多亏损预算越大，可买格数越多（复利探索）
    - 阈值设得更大（默认绝对回撤底线 20%），避免在网格低吸盈利期误杀
    """

    init_capital: float = 100000.0
    abs_floor_pct: float = 0.20      # 绝对回撤底线（vs 初始资金），触底前平滑收缩
    rel_floor_pct: float = 0.20      # 相对回撤底线（vs 高水位），两线取高者
    worst_drop: float = 0.20         # 持仓最坏跌幅假设（≈网格区间幅度）

    def __post_init__(self):
        self.peak = self.init_capital
        self.history = []      # (equity, max_grids_ratio) 轨迹
        self.floor_days = 0    # 格数被压到 0 的天数

    def update(self, equity: float) -> None:
        self.peak = max(self.peak, equity)
        self.history.append((equity, None))

    def _floor_value(self, equity: float) -> float:
        """底线资产：绝对底线与相对底线取高者（更保守）。"""
        abs_floor = self.init_capital * (1 - self.abs_floor_pct)
        rel_floor = self.peak * (1 - self.rel_floor_pct)
        return max(abs_floor, rel_floor)

    def allow_buy(self) -> bool:
        return True  # 冻结由 max_grids=0 平滑实现，无需硬开关

    def max_grids(self, grid_n: int, equity: float = None, price: float = None) -> int:
        """回撤预算反推的最大格数（事前控制）。

        亏损预算 = 当前资产 - 底线；可承受持仓 = 预算/最坏跌幅；
        再除以每格金额得最大格数。接近底线时平滑收缩至 0。
        """
        if equity is None:
            return grid_n
        floor_v = self._floor_value(equity)
        loss_budget = max(0.0, equity - floor_v)
        max_pos_value = loss_budget / max(self.worst_drop, 1e-6)
        per_lot_value = equity / max(grid_n, 1)
        mg = int(max_pos_value / max(per_lot_value, 1e-9))
        mg = max(0, min(mg, grid_n))
        if self.history:
            self.history[-1] = (equity, mg / grid_n)
        if mg == 0:
            self.floor_days += 1
        return mg

    def lot_size(self, equity: float, price: float, grid_n: int,
                 lot_unit: int = 100) -> int:
        """每格股数：盈利区按总资产折算（复利），亏损区封顶初始资金。"""
        base = equity if equity > self.init_capital else min(equity, self.init_capital)
        return max(int(base / max(grid_n, 1) / price / lot_unit), 1) * lot_unit

    def summary(self) -> dict:
        return {
            "peak": round(self.peak, 2),
            "floor_days": self.floor_days,
            "final_equity": round(self.history[-1][0], 2) if self.history else None,
        }
