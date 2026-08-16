"""网格回测环境：逐步仿真，供 RL（DQN）训练与评估。

步序（与回测引擎口径一致）：
    agent 在 t 日观察状态（收盘）做门控决策
    → 环境在 t+1 日开盘撮合网格目标仓位
    → t+1 日收盘后返回下一状态与奖赏

奖赏（差分 + 换手惩罚）：
    r = (ΔNAV_策略 - ΔNAV_基准) × 100 - β × 换手率%
差分奖赏使「拦截躺平」在上涨市被惩罚（必须跑赢买入持有基准才有正奖赏），
换手惩罚抑制过度交易。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import Account, CostConfig

# 持仓状态维度：持仓格数占比、浮动盈亏%（缩放后）
POS_DIM = 2


class GridBacktestEnv:
    """单股票网格环境。state = 市场特征 + 持仓状态，action = {0:拦截, 1:放行}。"""

    def __init__(
        self,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        grid_n: int = 10,
        range_pct: float = 0.20,
        capital: float = 100000.0,
        cost: Optional[CostConfig] = None,
        turn_penalty: float = 0.05,
    ):
        self.bars = bars.sort_values("date").reset_index(drop=True)
        # 特征按日期与 bars 对齐
        self.features = (
            features.set_index("date").reindex(self.bars["date"]).reset_index()
        )
        self.feat_cols = [c for c in self.features.columns if c != "date"]
        self.grid_n = grid_n
        self.capital0 = capital
        self.cost = cost or CostConfig()
        self.turn_penalty = turn_penalty

        p0 = float(self.bars["close"].iloc[0])
        self.lower = p0 * (1 - range_pct)
        self.upper = p0 * (1 + range_pct)
        self.grid_step = (self.upper - self.lower) / max(grid_n, 1)
        self.lot_shares = (
            max(int(capital / self.grid_n / p0 / 100), 1) * 100
        )
        self.state_dim = len(self.feat_cols) + POS_DIM
        self.reset()

    # ---- 基础 ----

    def reset(self) -> np.ndarray:
        self.t = 0
        self.account = Account(self.capital0, self.cost)
        self.nav_prev = self.capital0
        self.bench_prev = float(self.bars["close"].iloc[0])
        return self._state()

    def _grid_pos(self, price: float) -> int:
        """价格下方格数（与 GridStrategy.grid_pos 一致）。"""
        if price >= self.upper:
            return 0
        if price <= self.lower:
            return self.grid_n
        return int(np.clip(int((self.upper - price) / self.grid_step), 0, self.grid_n))

    def _unrealized_pct(self, close: float) -> float:
        if not self.account.lots:
            return 0.0
        cost_basis = sum(lot.shares * lot.buy_price for lot in self.account.lots)
        if cost_basis <= 0:
            return 0.0
        return (self.account.shares * close / cost_basis - 1) * 100

    def _state(self) -> np.ndarray:
        row = self.features.iloc[self.t]
        feat = np.nan_to_num(
            row[self.feat_cols].to_numpy(dtype=np.float32), nan=0.0
        )
        close = float(self.bars["close"].iloc[self.t])
        pos = np.array(
            [
                self.account.lots_count / self.grid_n,
                self._unrealized_pct(close) / 10.0,  # 缩放到常见范围
            ],
            dtype=np.float32,
        )
        return np.concatenate([feat, pos]).astype(np.float32)

    # ---- 核心交互 ----

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """t 日决策 → t+1 开盘撮合 → t+1 收盘返回（state, reward, done, info）。"""
        t, t1 = self.t, self.t + 1
        close_t = float(self.bars["close"].iloc[t])
        open_t1 = float(self.bars["open"].iloc[t1])
        day_t1 = self.bars["date"].iloc[t1]

        # 目标格数（与 GridStrategy 规则一致）
        target = self._grid_pos(close_t)
        if close_t < self.lower:  # 跌破下界冻结买入
            target = min(target, self.account.lots_count)
        if action == 0:  # 门控拦截：只卖不买
            target = min(target, self.account.lots_count)

        amount_traded = self._rebalance(target, open_t1, day_t1)

        close_t1 = float(self.bars["close"].iloc[t1])
        nav = self.account.cash + self.account.market_value(close_t1)

        # 差分奖赏：策略超额 - 换手惩罚
        r_strat = nav / self.nav_prev - 1
        r_bench = close_t1 / self.bench_prev - 1
        turnover = amount_traded / max(nav, 1e-9)
        reward = (r_strat - r_bench) * 100 - self.turn_penalty * turnover * 100

        self.nav_prev = nav
        self.bench_prev = close_t1
        self.t = t1
        done = self.t >= len(self.bars) - 1
        info = {
            "date": day_t1,
            "nav": nav,
            "shares": self.account.shares,
            "close": close_t1,
            "action": action,
        }
        return self._state(), float(reward), done, info

    def _rebalance(self, target: int, price: float, day) -> float:
        """撮合到目标格数，返回本步成交金额（换手用）。"""
        cur = self.account.lots_count
        amount = 0.0
        if target > cur:
            for _ in range(target - cur):
                before = self.account.cash
                if not self.account.buy(self.lot_shares, price, day):
                    break
                amount += before - self.account.cash
        elif target < cur:
            for _ in range(cur - target):
                before = self.account.cash
                if self.account.sell(self.lot_shares, price, day) == 0:
                    break
                amount += self.account.cash - before
        return amount
