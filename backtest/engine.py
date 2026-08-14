"""回测引擎：账户、成本、T+1 约束、逐日撮合与绩效统计。

撮合约定（无前视偏差）：
- 信号在 T 日收盘后产生（因子与模型输入仅用截至 T 日收盘的数据）
- T+1 日以开盘价撮合成交，滑点按比例偏移成交价
- A 股 T+1 制度：当日买入的批次当日不可卖出（FIFO 批次管理）
- 整手交易：买卖股数按 100 股向下取整
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A 股最小交易单位：1 手 = 100 股
LOT_SHARES = 100


@dataclass
class CostConfig:
    """交易成本模型。"""

    commission_rate: float = 2.5e-4   # 佣金（双边，万 2.5）
    min_commission: float = 5.0       # 单笔最低佣金（元）
    stamp_tax: float = 5e-4           # 印花税（仅卖出）
    slippage: float = 1e-3            # 滑点：成交价按比例偏移


@dataclass
class _Lot:
    """持仓批次（FIFO 卖出时逐批消耗）。"""

    shares: int
    buy_date: pd.Timestamp
    buy_price: float


class Account:
    """资金账户：现金 + FIFO 持仓批次，内置 T+1 约束与成本计算。"""

    def __init__(self, cash: float, cost: CostConfig):
        self.initial_cash = cash
        self.cash = cash
        self.cost = cost
        self.lots: List[_Lot] = []
        # 交易统计
        self.n_buy_trades = 0
        self.n_sell_trades = 0
        self.n_win_sells = 0
        self.n_lose_sells = 0
        self.total_cost = 0.0
        # 交易明细日志（每笔买卖一条）
        self.trades: List[Dict] = []

    # ---- 持仓 ----

    @property
    def shares(self) -> int:
        return sum(lot.shares for lot in self.lots)

    @property
    def lots_count(self) -> int:
        """占用批次数（近似「格数」，每批 1 手或多手）。"""
        return len(self.lots)

    def market_value(self, price: float) -> float:
        return self.shares * price

    # ---- 交易 ----

    def buy(self, shares: int, price: float, day: pd.Timestamp) -> bool:
        """按含滑点价买入，扣现金与佣金；现金不足返回 False。"""
        if shares <= 0:
            return False
        fill_price = price * (1 + self.cost.slippage)
        amount = shares * fill_price
        commission = max(amount * self.cost.commission_rate, self.cost.min_commission)
        if amount + commission > self.cash:
            return False

        self.cash -= amount + commission
        self.total_cost += commission
        self.lots.append(_Lot(shares=shares, buy_date=day, buy_price=fill_price))
        self.n_buy_trades += 1
        self.trades.append(
            {
                "date": day,
                "side": "buy",
                "shares": shares,
                "price": round(fill_price, 3),
                "amount": round(amount, 2),
                "fee": round(commission, 2),
                "cash_after": round(self.cash, 2),
            }
        )
        return True

    def sell(self, shares: int, price: float, day: pd.Timestamp) -> int:
        """FIFO 卖出（T+1：跳过买入日 >= 当日的批次），返回实际卖出股数。"""
        if shares <= 0:
            return 0
        fill_price = price * (1 - self.cost.slippage)
        remaining = shares
        sold = 0
        sell_fee = 0.0
        while remaining > 0 and self.lots:
            lot = self.lots[0]
            if lot.buy_date >= day:
                break  # T+1：该批次当日不可卖
            take = min(lot.shares, remaining)
            amount = take * fill_price
            commission = max(
                amount * self.cost.commission_rate, self.cost.min_commission
            )
            tax = amount * self.cost.stamp_tax

            self.cash += amount - commission - tax
            self.total_cost += commission + tax
            sell_fee += commission + tax

            # 胜负统计（与批次买价比较，已含滑点）
            if fill_price > lot.buy_price:
                self.n_win_sells += 1
            else:
                self.n_lose_sells += 1

            lot.shares -= take
            if lot.shares == 0:
                self.lots.pop(0)
            remaining -= take
            sold += take

        if sold > 0:
            self.n_sell_trades += 1
            self.trades.append(
                {
                    "date": day,
                    "side": "sell",
                    "shares": sold,
                    "price": round(fill_price, 3),
                    "amount": round(sold * fill_price, 2),
                    "fee": round(sell_fee, 2),
                    "cash_after": round(self.cash, 2),
                }
            )
        return sold


@dataclass
class BacktestResult:
    """单窗口回测结果。"""

    stats: Dict[str, float] = field(default_factory=dict)
    equity_curve: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None


def _perf_stats(
    account: Account,
    curve_total: np.ndarray,
    initial: float,
    n_days: int,
    buy_hold_return: float,
) -> Dict[str, float]:
    """基于每日总资产序列计算绩效指标。"""
    total_return = curve_total[-1] / initial - 1.0
    annual_return = (1.0 + total_return) ** (252.0 / max(n_days, 1)) - 1.0

    daily_ret = np.diff(curve_total) / curve_total[:-1]
    std = float(np.std(daily_ret)) if len(daily_ret) > 1 else 0.0
    sharpe = float(np.mean(daily_ret) / std * np.sqrt(252)) if std > 0 else 0.0

    cummax = np.maximum.accumulate(curve_total)
    drawdown = (cummax - curve_total) / cummax
    max_drawdown = float(np.max(drawdown)) if len(drawdown) else 0.0

    n_sells = account.n_win_sells + account.n_lose_sells
    win_rate = account.n_win_sells / n_sells if n_sells > 0 else 0.0

    return {
        "final_value": round(float(curve_total[-1]), 2),
        "total_return_pct": round(total_return * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe": round(sharpe, 2),
        "win_rate": round(win_rate, 4),
        "n_trades": account.n_buy_trades + account.n_sell_trades,
        "n_buy": account.n_buy_trades,
        "n_sell": account.n_sell_trades,
        "total_cost": round(account.total_cost, 2),
        "buy_hold_return_pct": round(buy_hold_return * 100, 2),
        "excess_pct": round((total_return - buy_hold_return) * 100, 2),
    }


def run_engine(
    bars: pd.DataFrame,
    signals: Optional[pd.DataFrame],
    strategy,
    capital: float = 100000.0,
    cost: Optional[CostConfig] = None,
    shares_per_lot: Optional[int] = None,
) -> BacktestResult:
    """逐日撮合回测。

    参数：
        bars          : 单股日线，按 date 升序，含 date/open/close
        signals       : 模型信号表（date/pred_reg/prob_up[/timing]）；
                        timing="next_open"（默认）：T 日信号在 T+1 开盘撮合；
                        timing="close"：T 日信号在 T 日收盘撮合（分钟级信号可用）；
                        某日缺信号时按无信号处理（策略自行决定门控行为）
        strategy      : 策略对象，需实现 target_lots(close, signal, cur_lots) -> int
        capital       : 初始资金
        cost          : 成本模型
        shares_per_lot: 每格固定股数；None 时按 capital/grid_n 首次买入价折算整手

    流程：第 i 日开盘执行 i-1 日的 next_open 信号，收盘执行 i 日的 close 信号，
    收盘后记录资金曲线；T+1 规则不变（当日买入批次当日不可卖）。
    """
    cost = cost or CostConfig()
    bars = bars.sort_values("date").reset_index(drop=True)
    if bars.empty or len(bars) < 2:
        return BacktestResult(stats={"error": "bars 不足"})

    # 信号按日期索引，便于查找（timing 缺省为 next_open，向后兼容）
    sig_map = {}
    if signals is not None and not signals.empty:
        has_timing = "timing" in signals.columns
        for _, r in signals.iterrows():
            sig_map[r["date"]] = {
                "pred_reg": float(r.get("pred_reg", 0.0)),
                "prob_up": float(r.get("prob_up", 0.5)),
                "timing": str(r["timing"]) if has_timing else "next_open",
            }

    # 网格以回测首日收盘价为基准
    strategy.setup(float(bars["close"].iloc[0]), capital)

    account = Account(capital, cost)
    lot_shares = shares_per_lot  # 延迟到首笔买入时初始化
    curve = []

    def _rebalance(ref_close: float, sig: Optional[dict], price: float, day) -> None:
        """按参考价与信号计算目标格数并撮合差额（day 供外部门控使用）。"""
        nonlocal lot_shares
        cur_lots = account.lots_count
        target = strategy.target_lots(ref_close, sig, cur_lots, day=day)

        if lot_shares is None:
            per_lot_cash = capital / max(strategy.grid_n, 1)
            lot_shares = max(int(per_lot_cash / price / LOT_SHARES), 1) * LOT_SHARES

        if target > cur_lots:
            # 逐格买入（现金不足时停止）
            for _ in range(target - cur_lots):
                if not account.buy(lot_shares, price, day):
                    break
        elif target < cur_lots:
            # 逐格卖出（T+1 限制下卖不掉的保留）
            for _ in range(cur_lots - target):
                if account.sell(lot_shares, price, day) == 0:
                    break

    for i in range(len(bars)):
        row = bars.iloc[i]
        day, open_px, close_px = row["date"], float(row["open"]), float(row["close"])

        # ---- 开盘：执行昨日 next_open 信号 ----
        if i > 0:
            prev_day = bars["date"].iloc[i - 1]
            prev_close = float(bars["close"].iloc[i - 1])
            sig = sig_map.get(prev_day)
            if sig is None or sig["timing"] == "next_open":
                _rebalance(prev_close, sig, open_px, day)

        # ---- 收盘：执行今日 close 信号 ----
        sig_today = sig_map.get(day)
        if sig_today is not None and sig_today["timing"] == "close":
            _rebalance(close_px, sig_today, close_px, day)

        # ---- 收盘：记录资金曲线 ----
        curve.append(
            {
                "date": day,
                "cash": account.cash,
                "shares": account.shares,
                "close": close_px,
                "market_value": account.market_value(close_px),
                "total": account.cash + account.market_value(close_px),
                "lots": account.lots_count,
            }
        )

    curve_df = pd.DataFrame(curve)
    total_arr = curve_df["total"].to_numpy(dtype=np.float64)

    # 买入持有基准：首日开盘买入、末日收盘估值（不计成本）
    buy_hold = float(bars["close"].iloc[-1] / bars["open"].iloc[0] - 1.0)

    stats = _perf_stats(account, total_arr, capital, len(bars) - 1, buy_hold)
    trades_df = pd.DataFrame(account.trades)
    if not trades_df.empty:
        # 追加持仓与总资产快照，展示每笔交易后的账户状态
        trades_df["lots_after"] = account_shares_snapshot(trades_df, curve_df)
    return BacktestResult(stats=stats, equity_curve=curve_df, trades=trades_df)


def account_shares_snapshot(
    trades_df: pd.DataFrame, curve_df: pd.DataFrame
) -> List[int]:
    """按交易日期从资金曲线取持仓股数快照（展示用）。"""
    shares_by_date = dict(zip(curve_df["date"], curve_df["shares"]))
    return [int(shares_by_date.get(d, 0)) for d in trades_df["date"]]
