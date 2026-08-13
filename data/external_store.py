"""外部数据存储层：基本面事件、宏观序列、图结构。

目录结构（cache/external/ 下）：

    cache/external/
    ├── fundamental/{code}.parquet   # 个股基本面事件长表（每行一个事件）
    ├── macro/{name}.parquet         # 宏观时间序列（如 fx_usdcny 美元兑人民币）
    └── graph/
        ├── nodes.parquet            # 图节点：stock / industry / macro
        └── edges.parquet            # 图边：belongs_to / correlates / affects

Schema 规范：

1. fundamental（基本面事件，可空字段按事件类型取舍）：
   date          : 公告/生效日（时间对齐基准，避免未来信息泄漏）
   code          : 股票代码
   event_type    : annual_report | semi_report | quarterly_report |
                   performance_forecast | dividend
   revenue / revenue_yoy / net_profit / net_profit_yoy : 财务数据（财报类事件）
   eps / roe     : 每股收益 / 净资产收益率（财报类事件）
   dividend_per_share : 每股分红（分红类事件）
   ex_date / pay_date : 除权除息日 / 派息日（分红类事件，可空）

2. macro（宏观序列）：date + 若干数值列，一个指标一个文件
   （如 fx_usdcny.parquet 仅含 date / usdcny 两列）

3. graph：
   nodes : node_id, node_type(stock|industry|macro), code, name
   edges : src, dst, edge_type(belongs_to|correlates|affects)

数据准备（拉取实现见 data/fetcher_external.py，CLI：python main.py fetch-external）：
- 基本面：baostock query_profit_data / query_dividend_data / 业绩预告 / 业绩快报
- 宏观利率：存款/贷款基准利率、存款准备金率、货币供应量（baostock）
- 汇率等需外部数据源（如 akshare），尚未接入
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)

# ---- 目录 ----
EXTERNAL_DIR = DATA_DIR / "external"
FUNDAMENTAL_DIR = EXTERNAL_DIR / "fundamental"
MACRO_DIR = EXTERNAL_DIR / "macro"
GRAPH_DIR = EXTERNAL_DIR / "graph"
for _d in (FUNDAMENTAL_DIR, MACRO_DIR, GRAPH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- 基本面事件 schema ----
FUNDAMENTAL_EVENT_TYPES: List[str] = [
    "annual_report",
    "semi_report",
    "quarterly_report",
    "performance_forecast",
    "dividend",
]

FUNDAMENTAL_COLUMNS: List[str] = [
    "date", "code", "event_type",
    "revenue", "revenue_yoy", "net_profit", "net_profit_yoy",
    "eps", "roe", "dividend_per_share", "ex_date", "pay_date",
]

FUNDAMENTAL_DTYPES = {
    "date": "datetime64[ns]",
    "code": "string",
    "event_type": "string",
    "revenue": "float64",
    "revenue_yoy": "float64",
    "net_profit": "float64",
    "net_profit_yoy": "float64",
    "eps": "float64",
    "roe": "float64",
    "dividend_per_share": "float64",
    "ex_date": "datetime64[ns]",
    "pay_date": "datetime64[ns]",
}

# ---- 图 schema ----
GRAPH_NODE_COLUMNS: List[str] = ["node_id", "node_type", "code", "name"]
GRAPH_EDGE_COLUMNS: List[str] = ["src", "dst", "edge_type"]


def fundamental_path(code: str) -> Path:
    safe = code.replace(".", "_")
    return FUNDAMENTAL_DIR / f"{safe}.parquet"


def list_fundamental_codes() -> List[str]:
    """列出已有基本面事件数据的股票代码。"""
    codes = []
    for f in sorted(FUNDAMENTAL_DIR.glob("*.parquet")):
        codes.append(f.stem.replace("_", "."))
    return codes


def save_fundamental_events(df: pd.DataFrame, code: str) -> None:
    """保存单只股票的基本面事件表（覆盖写，按 date 排序）。"""
    out = df.copy()
    out["code"] = code
    missing = [c for c in FUNDAMENTAL_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"基本面事件缺少字段: {missing}")
    out = out[FUNDAMENTAL_COLUMNS]
    for col, dtype in FUNDAMENTAL_DTYPES.items():
        out[col] = out[col].astype(dtype)
    out = out.sort_values("date").reset_index(drop=True)
    out.to_parquet(fundamental_path(code), index=False)


def load_fundamental_events(code: str) -> Optional[pd.DataFrame]:
    """读取单只股票的基本面事件表；不存在返回 None。"""
    path = fundamental_path(code)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def macro_path(name: str) -> Path:
    return MACRO_DIR / f"{name}.parquet"


def list_macro_names() -> List[str]:
    """列出已有宏观序列的指标名（如 fx_usdcny）。"""
    return sorted(f.stem for f in MACRO_DIR.glob("*.parquet"))


def save_macro_series(df: pd.DataFrame, name: str) -> None:
    """保存宏观时间序列。

    df 必须含 date 列与至少一个数值列，按 date 升序去重。
    """
    out = df.copy()
    if "date" not in out.columns:
        raise ValueError("宏观序列必须包含 date 列")
    out["date"] = pd.to_datetime(out["date"])
    value_cols = [c for c in out.columns if c != "date"]
    if not value_cols:
        raise ValueError("宏观序列至少需要一个数值列")
    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = (
        out[["date"] + value_cols]
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    out.to_parquet(macro_path(name), index=False)


def load_macro_series(name: str) -> Optional[pd.DataFrame]:
    """读取宏观序列（date + 数值列）；不存在返回 None。"""
    path = macro_path(name)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """保存图结构（节点 + 边），覆盖写。"""
    nodes = nodes_df.copy()
    edges = edges_df.copy()
    missing_n = [c for c in GRAPH_NODE_COLUMNS if c not in nodes.columns]
    missing_e = [c for c in GRAPH_EDGE_COLUMNS if c not in edges.columns]
    if missing_n or missing_e:
        raise ValueError(f"图节点缺少字段 {missing_n}，图边缺少字段 {missing_e}")

    nodes = nodes[GRAPH_NODE_COLUMNS].drop_duplicates(subset=["node_id"])
    nodes["node_id"] = nodes["node_id"].astype(str)
    nodes["node_type"] = nodes["node_type"].astype(str)
    nodes["code"] = nodes["code"].fillna("").astype(str)
    nodes["name"] = nodes["name"].fillna("").astype(str)

    edges = edges[GRAPH_EDGE_COLUMNS].drop_duplicates()
    edges["src"] = edges["src"].astype(str)
    edges["dst"] = edges["dst"].astype(str)
    edges["edge_type"] = edges["edge_type"].astype(str)

    # 校验：边的两端必须存在于节点表
    node_ids = set(nodes["node_id"])
    invalid = edges[~edges["src"].isin(node_ids) | ~edges["dst"].isin(node_ids)]
    if not invalid.empty:
        raise ValueError(f"图边引用了不存在的节点: {invalid.to_dict('records')}")

    nodes.to_parquet(GRAPH_DIR / "nodes.parquet", index=False)
    edges.to_parquet(GRAPH_DIR / "edges.parquet", index=False)
    logger.info(
        "图已保存：%d 个节点，%d 条边", len(nodes), len(edges)
    )


def load_graph() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取图结构，返回 (nodes, edges)；不存在时返回空表。"""
    nodes_path = GRAPH_DIR / "nodes.parquet"
    edges_path = GRAPH_DIR / "edges.parquet"
    nodes = pd.read_parquet(nodes_path) if nodes_path.exists() else pd.DataFrame(columns=GRAPH_NODE_COLUMNS)
    edges = pd.read_parquet(edges_path) if edges_path.exists() else pd.DataFrame(columns=GRAPH_EDGE_COLUMNS)
    return nodes, edges
