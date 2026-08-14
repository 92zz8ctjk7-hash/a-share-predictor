"""回测模块：网格交易引擎、策略、因子 IC 分析与多窗口编排。"""

from backtest.factor_eval import factor_ic_report
from backtest.run import run_backtest

__all__ = ["run_backtest", "factor_ic_report"]
