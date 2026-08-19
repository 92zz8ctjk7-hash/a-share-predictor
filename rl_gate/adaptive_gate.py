"""自适应分位数门控控制器：分位数阈值 + 探索-利用平衡（P1 方案）。

核心机制：
- 利用基线：θ = 近期 prob_up 分布的 (1-pass_rate) 分位数，自动跟随模型
  输出漂移（治固定 0.5 阈值与分布脱节导致的零交易问题）
- 探索机制：prob < θ 时以概率 ε 放行（ε-greedy），收集「模型说跌但
  实际如何」的真实反馈
- 元控制：探索买入 K 日后结算绝对收益（减成本），乘性权重更新 ε——
  亏损则指数收紧、盈利则缓慢恢复；连亏 max_consec_loss 笔触发熔断
  （ε 降至 ε_min 并冷却 cool_days 个交易日）
- 窗口自适应：核算/分位窗口 W 随已实现波动率反向调节
  （高波动缩短窗口加快适应）

引擎语义对齐：buy_gate(day) 在 T+1 开盘撮合时被调用（day=T+1），
决策依据为前一交易日 T 收盘的 prob_up，无前视。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdaptiveGateController:
    """逐日决策的门控控制器，状态跨段连续（每只股票一个实例）。"""

    def __init__(
        self,
        pass_rate: float = 0.40,
        quant_win: int = 60,
        eps_init: float = 0.10,
        eps_min: float = 0.02,
        eps_max: float = 0.20,
        eta: float = 0.30,
        horizon: int = 5,
        cool_days: int = 20,
        max_consec_loss: int = 3,
        cost_pct: float = 0.0025,
        warmup: int = 30,
        seed: int = 42,
    ):
        self.pass_rate = pass_rate
        self.quant_win = quant_win
        self.eps_init = eps_init
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eta = eta
        self.horizon = horizon
        self.cool_days = cool_days
        self.max_consec_loss = max_consec_loss
        self.cost_pct = cost_pct
        self.warmup = warmup
        self.rng = np.random.default_rng(seed)

        # 状态（bind 时重置）
        self.days: List[pd.Timestamp] = []
        self.day_idx: Dict[pd.Timestamp, int] = {}
        self.open = np.array([])
        self.close = np.array([])
        self.prob_by_date: Dict[pd.Timestamp, float] = {}
        self._reset_state()

    def _reset_state(self) -> None:
        self.prob_hist: List[float] = []
        self.eps = self.eps_init
        self.explore_buys: List[List] = []  # [exec_idx, buy_price]
        self.consec_loss = 0
        self.cooldown_until = -1
        self.n_block = 0
        self.n_explore = 0
        self.n_explore_win = 0
        self.n_settled = 0
        self.theta_last: Optional[float] = None

    def bind(
        self,
        trade_days,
        open_px,
        close_px,
        prob_by_date: Optional[Dict] = None,
    ) -> None:
        """绑定交易日历与行情；prob_by_date 可后续通过 add_probs 增量补充。"""
        self.days = [pd.Timestamp(d) for d in trade_days]
        self.day_idx = {d: i for i, d in enumerate(self.days)}
        self.open = np.asarray(open_px, dtype=np.float64)
        self.close = np.asarray(close_px, dtype=np.float64)
        self.prob_by_date = {}
        if prob_by_date:
            self.add_probs(prob_by_date)
        self._reset_state()

    def add_probs(self, prob_by_date: Dict) -> None:
        """增量补充信号（每段 walk 训练完成后注入该段 prob）。"""
        for k, v in prob_by_date.items():
            self.prob_by_date[pd.Timestamp(k)] = float(v)

    # ---- 引擎入口：T+1 开盘撮合时调用 ----

    def gate(self, day) -> bool:
        day = pd.Timestamp(day)
        e = self.day_idx.get(day)
        if e is None or e == 0:
            return True

        # 1. 结算已到期的探索买入（更新 ε / 熔断状态）
        self._settle(e)

        # 2. 决策日 = 前一交易日，取其收盘 prob
        d_prev = self.days[e - 1]
        prob = self.prob_by_date.get(d_prev)
        if prob is None:
            return True  # 无信号日放行（与引擎无信号语义一致）
        self.prob_hist.append(float(prob))

        # 3. 预热期：概率历史不足，放行收集
        if len(self.prob_hist) < self.warmup:
            return True

        # 4. 分位数阈值（自适应窗口）
        w = self._adaptive_window(e)
        theta = float(np.quantile(self.prob_hist[-w:], 1.0 - self.pass_rate))
        self.theta_last = theta
        if prob >= theta:
            return True  # 利用：放行

        # 5. 阈值下：冷却期拦截，否则按 ε 探索
        if e <= self.cooldown_until:
            self.n_block += 1
            return False
        if self.rng.random() < self.eps:
            self.n_explore += 1
            self.explore_buys.append([e, float(self.open[e])])
            return True
        self.n_block += 1
        return False

    # ---- 内部机制 ----

    def _settle(self, e: int) -> None:
        still = []
        for idx, p0 in self.explore_buys:
            tgt = idx + self.horizon
            if tgt <= e and tgt < len(self.days):
                r = float(self.close[tgt] / p0 - 1 - self.cost_pct)
                self._update_eps(r, e)
            else:
                still.append([idx, p0])
        self.explore_buys = still

    def _update_eps(self, r: float, e: int) -> None:
        self.n_settled += 1
        if r > 0:
            self.n_explore_win += 1
        r_pct = float(np.clip(r * 100, -5, 5))
        self.eps = float(np.clip(
            self.eps * np.exp(self.eta * r_pct), self.eps_min, self.eps_max
        ))
        if r < 0:
            self.consec_loss += 1
            if self.consec_loss >= self.max_consec_loss:
                # 熔断：ε 降至下限并冷却
                self.eps = self.eps_min
                self.cooldown_until = e + self.cool_days
                self.consec_loss = 0
                logger.info(
                    "自适应门控熔断：探索连亏 %d 笔，冷却至 %s",
                    self.max_consec_loss,
                    self.days[min(e + self.cool_days, len(self.days) - 1)].date(),
                )
        else:
            self.consec_loss = 0

    def _adaptive_window(self, e: int) -> int:
        """分位窗口随已实现波动率反向调节：高波动缩短（加快适应）。"""
        if e < 60:
            return self.quant_win
        c = self.close[max(0, e - 250): e + 1]
        ret = np.diff(c) / c[:-1]
        if len(ret) < 40:
            return self.quant_win
        sigma_recent = float(np.std(ret[-20:]))
        # 历史窗口波动中位数（60 日窗口、20 日步进）
        sigs = [
            float(np.std(ret[i: i + 60]))
            for i in range(0, max(1, len(ret) - 60), 20)
        ]
        sigma_base = float(np.median(sigs)) if sigs else sigma_recent
        if sigma_recent <= 1e-8:
            return self.quant_win
        w = int(self.quant_win * sigma_base / sigma_recent)
        return int(np.clip(w, 20, 120))

    def summary(self) -> Dict:
        return {
            "n_block": self.n_block,
            "n_explore": self.n_explore,
            "n_settled": self.n_settled,
            "explore_win_rate": (
                self.n_explore_win / self.n_settled if self.n_settled else 0.0
            ),
            "eps_final": round(self.eps, 4),
            "theta_final": round(self.theta_last, 4) if self.theta_last else None,
        }
