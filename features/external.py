"""外部数据特征化：事件表/宏观序列 → 每日状态特征。

三类外部数据的特征化与注入辅助：
- 基本面事件（财报/业绩/分红）→ 个股每日状态特征（分钟模型 static 输入）
- 宏观序列（汇率等）→ 每日宏观特征（基座样本与分钟样本均可 join）
- 图结构 → 一阶邻居聚合特征（基座模型静态输入）

无泄漏约定：
- 基本面只用「公告日 ≤ 信息截止日」的事件（merge_asof backward）
- 盘前截面（分钟样本 10:00）统一用前一交易日的特征（align_features_previous_day）
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from data import external_store
from data.graph_store import GraphStore

logger = logging.getLogger(__name__)

# 财报类事件（用于 days_since_report / 最近财务指标）
_FIN_EVENT_TYPES = [
    "annual_report", "semi_report", "quarterly_report", "performance_forecast",
]


def compute_fundamental_features(
    events_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    close_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """把基本面事件表转成每日状态特征，index=交易日。

    事件按公告日对齐（只用公告日 ≤ 当日的财务数据）；
    ex_date（除权日）属于已公告计划，days_to_ex_div 用 forward 查询。

    返回列：
        days_since_report : 距最近财报公告天数（NaN=尚无财报）
        eps_last / roe_last / revenue_yoy_last / net_profit_yoy_last
        div_ps_ttm        : 近 365 天每股分红累计（元）
        div_yield_ttm     : 近 365 天分红率（%，需 close_series）
        days_to_ex_div    : 距下一个已公告除权日天数（NaN=无已知计划）
        days_since_ex_div : 距最近除权日天数
    """
    events = events_df[events_df["date"].notna()].sort_values("date")
    # 特征日期 = 交易日 ∪ 事件公告日（保证公告信息自公告日起可查）
    all_dates = pd.DatetimeIndex(dates).union(pd.DatetimeIndex(events["date"]))
    base = pd.DataFrame({"date": all_dates}).sort_values("date")
    feats = base.copy()

    # 1. 财务事件：公告日 ≤ 当日的最近一条
    fin = events[events["event_type"].isin(_FIN_EVENT_TYPES)]
    if not fin.empty:
        fin = fin.rename(columns={"date": "event_date"})[
            ["event_date", "eps", "roe", "revenue_yoy", "net_profit_yoy"]
        ]
        merged = pd.merge_asof(
            base, fin, left_on="date", right_on="event_date",
            direction="backward",
        )
        feats["days_since_report"] = (merged["date"] - merged["event_date"]).dt.days
        for col in ("eps", "roe", "revenue_yoy", "net_profit_yoy"):
            feats[f"{col}_last"] = merged[col]

    # 2. 分红事件（按除权日 ex_date 对齐）
    div = events[events["event_type"] == "dividend"].copy()
    div["ex_d"] = pd.to_datetime(div["ex_date"])
    div = div[div["ex_d"].notna() & div["dividend_per_share"].notna()]
    if not div.empty:
        div = div.sort_values("ex_d")

        fwd = pd.merge_asof(
            base, div[["ex_d"]], left_on="date", right_on="ex_d",
            direction="forward",
        )
        feats["days_to_ex_div"] = (fwd["ex_d"] - fwd["date"]).dt.days

        bwd = pd.merge_asof(
            base, div[["ex_d"]], left_on="date", right_on="ex_d",
            direction="backward",
        )
        feats["days_since_ex_div"] = (bwd["date"] - bwd["ex_d"]).dt.days

        # 近 365 天每股分红累计
        cross = base.merge(div[["ex_d", "dividend_per_share"]], how="cross")
        cross = cross[
            (cross["ex_d"] <= cross["date"])
            & (cross["ex_d"] > cross["date"] - pd.Timedelta(days=365))
        ]
        ttm = cross.groupby("date")["dividend_per_share"].sum()
        feats["div_ps_ttm"] = feats["date"].map(ttm).fillna(0.0)

        if close_series is not None:
            close = close_series.reindex(feats["date"])
            feats["div_yield_ttm"] = feats["div_ps_ttm"] / close.values * 100.0

    feats = feats.set_index("date")
    # 状态类特征向前填充（最近已知值持续有效）
    state_cols = [c for c in feats.columns if c.startswith(("days_", "eps_", "roe_", "revenue_", "net_profit_"))]
    feats[state_cols] = feats[state_cols].ffill()
    return feats


def compute_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """宏观序列 → 每日特征，index=交易日。

    macro_df 含 date 列与数值列，对每个数值列生成：
        {col}           : 原值
        {col}_ret_1d/5d/20d : N 日变化率（%）
        {col}_ma20_bias : 相对 20 日均线偏离（%）
    """
    out = macro_df.set_index("date").sort_index()
    res = pd.DataFrame(index=out.index)
    for col in out.columns:
        s = out[col].astype(float)
        res[col] = s
        res[f"{col}_ret_1d"] = s.pct_change(1) * 100.0
        res[f"{col}_ret_5d"] = s.pct_change(5) * 100.0
        res[f"{col}_ret_20d"] = s.pct_change(20) * 100.0
        res[f"{col}_ma20_bias"] = (s / s.rolling(20).mean() - 1) * 100.0
    return res.ffill()


def merge_asof_previous_day(sample: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """把 index=日期 的特征表按「sample.date 的前一日历日」backward 合并。

    用于盘前截面注入（分钟样本 10:00）：样本交易日 T 只能使用
    T-1 日历日收盘时已知的信息（含周末公告、工作日汇率）。
    返回 sample 加特征列。
    """
    feat_flat = feats.copy().reset_index()
    idx_col = feat_flat.columns[0]
    feat_flat = feat_flat.rename(columns={idx_col: "_fdate"})

    s = sample.copy().sort_values("date").reset_index(drop=True)
    s["_join"] = s["date"] - pd.Timedelta(days=1)
    merged = pd.merge_asof(
        s, feat_flat.sort_values("_fdate"),
        left_on="_join", right_on="_fdate", direction="backward",
    )
    return merged.drop(columns=["_join", "_fdate"]).sort_index()


def detect_external() -> Dict[str, bool]:
    """自动检测外部数据就绪情况，返回 {'fundamental': bool, 'macro': bool, 'graph': bool}。"""
    graph = GraphStore.from_disk()
    ready = {
        "fundamental": len(external_store.list_fundamental_codes()) > 0,
        "macro": len(external_store.list_macro_names()) > 0,
        "graph": not graph.is_empty,
    }
    return ready


def load_macro_features() -> Optional[pd.DataFrame]:
    """加载全部宏观序列并特征化，按 date 外连接合并，index=交易日。"""
    names = external_store.list_macro_names()
    if not names:
        return None
    frames = []
    for name in names:
        df = external_store.load_macro_series(name)
        if df is not None and not df.empty:
            frames.append(compute_macro_features(df))
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    return out.sort_index().ffill()


def build_node_features() -> pd.DataFrame:
    """构造图节点特征表（index=node_id）。

    当前从宏观序列自动为 macro 节点生成特征（最新值与 20 日变化率）；
    industry 等节点的特征可后续扩展（如行业指数收益）。
    """
    rows = {}
    for name in external_store.list_macro_names():
        df = external_store.load_macro_series(name)
        if df is None or df.empty:
            continue
        s = df.set_index("date").iloc[:, 0].astype(float)
        rows[f"macro_{name}"] = {
            "latest": float(s.iloc[-1]),
            "ret_20d": float(s.iloc[-1] / s.iloc[-21] - 1) if len(s) > 21 else 0.0,
        }
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(rows, orient="index")


def load_graph_features() -> Dict[str, pd.Series]:
    """对每个股票节点聚合一阶邻居特征，返回 {stock_code: 图特征 Series}。

    图特征列名 g_{edge_type}_{col}（如 g_affects_latest）。
    """
    graph = GraphStore.from_disk()
    if graph.is_empty:
        return {}
    node_features = build_node_features()
    if node_features.empty:
        logger.warning("图存在但无可用的节点特征（macro 数据缺失），跳过图特征")
        return {}

    out: Dict[str, pd.Series] = {}
    for node_id in graph.node_ids("stock"):
        feats = graph.aggregate(node_id, node_features)
        if not feats.empty:
            out[node_id] = feats
    return out
