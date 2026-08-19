"""影子账户：生产策略的模拟记账（现金 + FIFO 筹码 + 盈亏跟踪）。

与回测引擎同撮合语义：T 日收盘决策 → T+1 开盘价成交、T+1 限制、
整手交易、佣金/印花税/滑点。每日 serve 时先执行昨日决策，再生成今日决策，
推送账户状态与具体波段计划（买卖股数/金额/目标价位）。

持久化：cache/signals/shadow_account.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)

ACCOUNT_PATH = DATA_DIR / "signals" / "shadow_account.json"


def account_path(name: str):
    """账户持久化路径：shadow=影子账户（历史兼容路径），其他名独立文件。"""
    if name == "shadow":
        return ACCOUNT_PATH
    return DATA_DIR / "signals" / f"account_{name}.json"

# 与回测一致的成本参数
COMMISSION_RATE = 2.5e-4
MIN_COMMISSION = 5.0
STAMP_TAX = 5e-4
SLIPPAGE = 1e-3
LOT_UNIT = 100


@dataclass
class Lot:
    shares: int
    buy_date: str
    buy_price: float


@dataclass
class ShadowAccount:
    init_capital: float = 100000.0
    cash: float = 100000.0
    lots: List[Lot] = field(default_factory=list)
    lot_shares: Optional[int] = None       # 首笔买入时锁定（与回测静态每格一致）
    grid_anchor: Optional[float] = None    # 网格锚定价（建仓日收盘，清仓后重置）
    anchor_date: Optional[str] = None
    realized_pnl: float = 0.0
    n_win: int = 0
    n_lose: int = 0
    pending: Optional[Dict] = None         # 待执行的昨日决策
    last_update: Optional[str] = None
    name: str = "shadow"                   # 账户名（shadow/real）
    prev_equity: Optional[float] = None    # 上一记录日总资产（分日收益）
    prev_equity_date: Optional[str] = None

    # ---- 持仓 ----

    @property
    def shares(self) -> int:
        return sum(l.shares for l in self.lots)

    def equity(self, price: float) -> float:
        return self.cash + self.shares * price

    def avg_cost(self) -> Optional[float]:
        if not self.lots:
            return None
        total = sum(l.shares * l.buy_price for l in self.lots)
        return total / self.shares

    # ---- 交易 ----

    def _ensure_lot_size(self, price: float) -> int:
        if self.lot_shares is None:
            per_lot_cash = self.init_capital / 10  # grid_n=10
            self.lot_shares = max(int(per_lot_cash / price / LOT_UNIT), 1) * LOT_UNIT
        return self.lot_shares

    def buy(self, n_lots: int, price: float, day: str) -> int:
        """按含滑点价买入 n_lots 格，现金不足时部分成交，返回实际格数。"""
        lot = self._ensure_lot_size(price)
        fill_price = price * (1 + SLIPPAGE)
        bought = 0
        for _ in range(n_lots):
            amount = lot * fill_price
            commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
            if amount + commission > self.cash:
                break
            self.cash -= amount + commission
            self.lots.append(Lot(shares=lot, buy_date=day, buy_price=fill_price))
            bought += 1
        return bought

    def sell(self, n_lots: int, price: float, day: str, sell_all: bool = False) -> int:
        """FIFO 卖出 n_lots 格（sell_all 全部），遵守 T+1，返回实际格数。"""
        lot = self.lot_shares or self._ensure_lot_size(price)
        fill_price = price * (1 - SLIPPAGE)
        sold = 0
        target_lots = len(self.lots) if sell_all else n_lots
        while sold < target_lots and self.lots:
            head = self.lots[0]
            if head.buy_date >= day:  # T+1：当日买入不可卖
                break
            take = min(lot, head.shares)
            amount = take * fill_price
            commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
            tax = amount * STAMP_TAX
            pnl = (fill_price - head.buy_price) * take - commission - tax
            self.cash += amount - commission - tax
            self.realized_pnl += pnl
            if pnl > 0:
                self.n_win += 1
            else:
                self.n_lose += 1
            head.shares -= take
            if head.shares == 0:
                self.lots.pop(0)
            sold += 1
        return sold

    # ---- 持久化 ----

    def save(self) -> None:
        p = account_path(self.name)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, name: str = "shadow") -> "ShadowAccount":
        p = account_path(name)
        if not p.exists():
            return cls(name=name)
        data = json.loads(p.read_text())
        lots = [Lot(**l) for l in data.pop("lots", [])]
        data.setdefault("name", name)
        acc = cls(**data)
        acc.lots = lots
        return acc

    # ---- 状态摘要 ----

    def snapshot(self, price: float) -> Dict:
        eq = self.equity(price)
        ac = self.avg_cost()
        unrealized = (price / ac - 1) * 100 if ac else 0.0
        n_sells = self.n_win + self.n_lose
        # 分日收益：与上一记录日总资产对比
        daily_pct = None
        if self.prev_equity and self.prev_equity > 0:
            daily_pct = round((eq / self.prev_equity - 1) * 100, 2)
        return {
            "name": self.name,
            "equity": round(eq, 0),
            "cash": round(self.cash, 0),
            "position_value": round(self.shares * price, 0),
            "position_pct": round(self.shares * price / eq * 100, 1) if eq > 0 else 0.0,
            "shares": self.shares,
            "n_lots": len(self.lots),
            "avg_cost": round(ac, 3) if ac else None,
            "unrealized_pct": round(unrealized, 2),
            "realized_pnl": round(self.realized_pnl, 0),
            "total_return_pct": round((eq / self.init_capital - 1) * 100, 2),
            "daily_return_pct": daily_pct,
            "win_rate": round(self.n_win / n_sells * 100, 1) if n_sells else None,
            "grid_anchor": self.grid_anchor,
        }
