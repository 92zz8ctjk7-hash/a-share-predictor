"""P2 RL 强度控制器：用轨迹解码器的 σ（不确定度）控制仓位强度。

基于 P1 的发现：方向 μ 接近随机游走不可预测，但 σ（不确定度）有真实日内结构。
因此 P2 不用 μ 预测方向，而是让 RL 学习「σ 多大时该用多大仓位强度」：
    σ 大（不确定）→ 轻仓/谨慎；σ 小（确定）→ 正常仓位

动作空间：仓位强度 w ∈ {0, 0.25, 0.5, 0.75, 1.0}（5 值离散）
    目标仓位 = 网格自然目标格数 × w
奖赏：差分超额收益（复用 GridBacktestEnv 口径）
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from rl_gate.env import GridBacktestEnv, POS_DIM
from rl_gate.opportunity import build_day_features
from rl_gate.trajectory import (build_window_returns,
                                 build_window_trajectory_samples,
                                 train_trajectory)

logger = logging.getLogger(__name__)

# 仓位强度档位（动作 0-4 → 强度系数）
INTENSITY_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_ACTIONS = len(INTENSITY_LEVELS)


def build_daily_sigma(
    min_bars: pd.DataFrame, base_preds, code: str, epochs: int = 60,
    lookback: int = 3,
) -> Dict:
    """训练 30 分钟窗口轨迹解码器，返回每日 σ_day。

    方向 A：取当日第一个可预测窗口（开盘窗，win_idx=lookback）的 σ，
    不做日平均——保留日内结构，与 10:00 决策点天然对齐。
    返回 {date: sigma_day}，供 IntensityGridEnv 作为状态特征。
    """
    win_ret = build_window_returns(min_bars)
    r_seq, cond, target, win_idx, meta = build_window_trajectory_samples(
        win_ret, base_preds, code, lookback=lookback
    )
    model, _ = train_trajectory(r_seq, cond, target, win_idx, epochs=epochs)
    model.eval()

    import torch

    with torch.no_grad():
        _, ls = model(
            torch.tensor(r_seq), torch.tensor(cond),
            torch.tensor(win_idx, dtype=torch.long),
        )
    sig = np.exp(ls.numpy())
    meta = meta.copy()
    meta["sigma"] = sig
    # 取每日第一个可预测窗口（最小 win_idx）的 σ，保留日内结构
    daily_sigma = meta.groupby("date").apply(
        lambda g: g.loc[g["win_idx"].idxmin(), "sigma"], include_groups=False
    )
    return daily_sigma.to_dict()


def build_realized_sigma(bars: pd.DataFrame, window: int = 20) -> Dict:
    """历史已实现波动率：过去 window 日收益的标准差（方向 B）。

    不依赖 NLL 轨迹模型，天然有日度判别力（高波动日 σ 大），
    只用历史数据无未来信息。返回 {date: sigma_realized}。
    """
    df = bars.sort_values("date").reset_index(drop=True)
    ret = df["close"].pct_change()
    sigma = ret.rolling(window).std()
    return dict(zip(df["date"], sigma))


class IntensityGridEnv(GridBacktestEnv):
    """强度动作网格环境：action ∈ {0..4} → 强度 w，目标仓位 = 网格目标 × w。

    状态在 GridBacktestEnv 基础上追加 1 维 σ_day（缩放后）。
    """

    def __init__(self, bars, features, sigma_day: Dict, default_sigma: float = 0.005,
                 **kwargs):
        # 必须在 super().__init__ 之前设置（父类构造会调 reset→_state→_sigma_at）
        self.sigma_day = sigma_day
        self.default_sigma = default_sigma
        super().__init__(bars, features, **kwargs)
        self.state_dim = self.state_dim + 1  # 追加 σ_day
        self.n_actions = N_ACTIONS  # 供 train_dqn 推断动作数

    def _sigma_at(self, t: int) -> float:
        day = self.bars["date"].iloc[t]
        v = self.sigma_day.get(day, self.default_sigma)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return self.default_sigma
        return float(v)

    def _state(self) -> np.ndarray:
        base = super()._state()
        sigma = self._sigma_at(self.t)
        # σ 缩放（典型 σ_day≈0.005，除以 0.01 映射到 ~0.5 量级）
        sig_feat = np.array([sigma / 0.01], dtype=np.float32)
        return np.concatenate([base, sig_feat]).astype(np.float32)

    def step(self, action: int):
        """action = 强度档位 0-4 → w；目标仓位 = 网格自然目标 × w。"""
        t, t1 = self.t, self.t + 1
        close_t = float(self.bars["close"].iloc[t])
        open_t1 = float(self.bars["open"].iloc[t1])
        day_t1 = self.bars["date"].iloc[t1]

        w = INTENSITY_LEVELS[action]
        target_base = self._grid_pos(close_t)
        if close_t < self.lower:  # 跌破下界冻结买入
            target_base = min(target_base, self.account.lots_count)
        target = int(round(target_base * w))

        amount_traded = self._rebalance(target, open_t1, day_t1)

        close_t1 = float(self.bars["close"].iloc[t1])
        nav = self.account.cash + self.account.market_value(close_t1)

        r_strat = nav / self.nav_prev - 1
        r_bench = close_t1 / self.bench_prev - 1
        turnover = amount_traded / max(nav, 1e-9)
        reward = (r_strat - r_bench) * 100 - self.turn_penalty * turnover * 100

        self.nav_prev = nav
        self.bench_prev = close_t1
        self.t = t1
        done = self.t >= len(self.bars) - 1
        info = {"date": day_t1, "nav": nav, "shares": self.account.shares,
                "close": close_t1, "action": action, "intensity": w}
        return self._state(), float(reward), done, info


def run_hybrid_regime_episode(
    intensity_env: "IntensityGridEnv",
    intensity_agent,
    regime_by_date: Dict,
):
    """regime 混合执行：趋势市满仓拿收益，震荡市用强度 DQN 控仓降回撤。

    趋势日 → 满强度（满仓网格，抓趋势收益）；
    震荡日 → 强度 DQN 决策（高波动轻仓，控回撤）。
    返回 (curve DataFrame, env)。
    """
    s = intensity_env.reset()
    done = False
    rows = []
    while not done:
        day = intensity_env.bars["date"].iloc[intensity_env.t]
        is_trend = bool(regime_by_date.get(day, False))
        if is_trend:
            a = N_ACTIONS - 1  # 满强度 = 满仓网格
        else:
            a = intensity_agent.act(s, explore=False)
        s, r, done, info = intensity_env.step(a)
        info["regime"] = "trend" if is_trend else "range"
        rows.append(info)
    return pd.DataFrame(rows), intensity_env


def build_regime_map(
    idx_bars: pd.DataFrame,
    stock_bars: Optional[pd.DataFrame] = None,
    strength_th: float = 0.03,
    consistency_th: float = 0.6,
    use_stock_align: bool = True,
) -> Dict:
    """多维趋势判断：趋势强度 + 方向一致性 + （可选）个股与指数同向。

    is_trend = (|指数20日涨跌| > strength_th)
               AND (方向一致性 > consistency_th)
               AND （可选）个股与指数 20 日方向同向

    相比单指标（仅 |idx_ret_20d|）：方向一致性能区分「真趋势」（持续同向）
    与「伪趋势」（单日大波动但方向反复），减少牛市段误判为震荡。
    返回 {date: is_trend}。
    """
    idx = idx_bars.sort_values("date").set_index("date")
    close = idx["close"]
    ret = close.pct_change()
    ret20 = close.pct_change(20)

    # 维 1：趋势强度
    strength = ret20.abs()

    # 维 2：方向一致性（近 20 日中与 20 日趋势同向的天数占比）
    ret20_sign = np.sign(ret20)
    same_dir = (np.sign(ret) == ret20_sign).astype(float)
    consistency = same_dir.rolling(20).mean()

    is_trend = (strength > strength_th) & (consistency > consistency_th)

    # 维 3：个股与指数 20 日方向同向（可选）
    if use_stock_align and stock_bars is not None:
        stk = stock_bars.sort_values("date").set_index("date")
        stk_ret20_sign = np.sign(stk["close"].pct_change(20))
        aligned = stk_ret20_sign.reindex(is_trend.index)
        is_trend = is_trend & (aligned == ret20_sign)

    return is_trend.fillna(False).to_dict()


def build_tristate_regime(
    idx_bars: pd.DataFrame,
    stock_bars: Optional[pd.DataFrame] = None,
    strength_th: float = 0.03,
    consistency_th: float = 0.55,
) -> Dict:
    """三态趋势判断（区分方向）：返回 {date: 'up'|'down'|'range'}。

    上涨趋势：idx_ret_20d > +th 且方向一致性高（满仓抓趋势）
    下跌趋势：idx_ret_20d < -th（减仓避险，避免下跌满仓）
    震荡：|idx_ret_20d| <= th（用强度 DQN）

    相比二态（仅强度不分方向）：避免把下跌趋势误判为趋势而满仓大亏。
    """
    idx = idx_bars.sort_values("date").set_index("date")
    close = idx["close"]
    ret = close.pct_change()
    ret20 = close.pct_change(20)

    strength = ret20.abs()
    ret20_sign = np.sign(ret20)
    same_dir = (np.sign(ret) == ret20_sign).astype(float)
    consistency = same_dir.rolling(20).mean()

    regime = pd.Series("range", index=idx.index)
    # 上涨趋势：强度够 + 方向一致
    up = (ret20 > strength_th) & (consistency > consistency_th)
    # 下跌趋势：强度够（下跌）
    down = ret20 < -strength_th
    regime[up] = "up"
    regime[down] = "down"

    # 个股与指数同向才确认趋势（可选）
    if stock_bars is not None:
        stk = stock_bars.sort_values("date").set_index("date")
        stk_ret20_sign = np.sign(stk["close"].pct_change(20)).reindex(regime.index)
        misaligned = stk_ret20_sign != ret20_sign
        regime[misaligned & (regime != "range")] = "range"

    return regime.fillna("range").to_dict()


def run_tristate_regime_episode(
    intensity_env: "IntensityGridEnv",
    intensity_agent,
    regime_by_date: Dict,
    down_intensity: int = 1,
):
    """三态 regime 执行：上涨满仓、下跌减仓、震荡用强度 DQN。

    上涨日 → 满强度（满仓网格，抓趋势收益）；
    下跌日 → 低强度（减仓避险，避免下跌满仓）；
    震荡日 → 强度 DQN 决策（震荡优势）。
    """
    s = intensity_env.reset()
    done = False
    rows = []
    while not done:
        day = intensity_env.bars["date"].iloc[intensity_env.t]
        regime = regime_by_date.get(day, "range")
        if regime == "up":
            a = N_ACTIONS - 1  # 满强度
        elif regime == "down":
            a = down_intensity  # 低强度减仓
        else:
            a = intensity_agent.act(s, explore=False)  # 震荡用 DQN
        s, r, done, info = intensity_env.step(a)
        info["regime"] = regime
        rows.append(info)
    return pd.DataFrame(rows), intensity_env
