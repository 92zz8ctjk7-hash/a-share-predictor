"""回测结果绘图：资金曲线（对比买入持有）+ 仓位变化。

通用命令：
    python main.py plot-equity --code sz.000100 --window 1y

依赖 backtest 生成的 cache/meta/equity_{code}_{window}.csv；
回测时保存资金曲线后会自动调用 plot_equity 出图。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)


def plot_equity(code: str, window: str) -> Path:
    """绘制指定股票/窗口的资金曲线图，返回 png 路径。

    上图：策略总资产 vs 买入持有基准；下图：每日持仓股数（仓位节奏）。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    csv_path = DATA_DIR / "meta" / f"equity_{code}_{window}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"资金曲线不存在: {csv_path}，请先运行 backtest（默认保存资金曲线）"
        )

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    init = float(df["total"].iloc[0])
    final = float(df["total"].iloc[-1])
    total_ret = (final / init - 1) * 100
    buy_hold_ret = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    buy_hold = init * df["close"] / df["close"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    # 上图：资金曲线
    ax1.plot(df["date"], df["total"], color="#d62728", lw=1.8,
             label="网格策略（模型门控）")
    ax1.plot(df["date"], buy_hold, color="#7f7f7f", lw=1.2, ls="--",
             label="买入持有")
    ax1.axhline(init, color="#bbbbbb", lw=0.8, ls=":")
    ax1.fill_between(df["date"], init, df["total"],
                     where=df["total"] >= init, alpha=0.08, color="#d62728")
    ax1.set_title(
        f"{code} 回测资金曲线 · 窗口 {window}"
        f"（初始 {init:,.0f} 元 | 策略 {total_ret:+.2f}% vs 买入持有 {buy_hold_ret:+.2f}%）",
        fontsize=12,
    )
    ax1.set_ylabel("总资产（元）")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # 下图：持仓股数（仓位节奏）
    ax2.fill_between(df["date"], df["shares"], 0, color="#1f77b4",
                     alpha=0.5, step="mid")
    ax2.set_ylabel("持仓（股）")
    ax2.set_xlabel("日期")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = csv_path.with_suffix(".png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("资金曲线图已保存到 %s", out)
    return out
