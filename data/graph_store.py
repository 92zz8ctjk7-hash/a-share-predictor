"""图结构抽象层（纯 pandas 实现，无额外依赖）。

以「节点/边」组织股票、行业与宏观指标之间的关联，提供一阶邻居
查询与特征聚合，聚合结果可作为模型的图特征输入。

节点约定：
- stock 节点：node_id = 股票代码（如 sh.600000），code 同 node_id
- industry 节点：node_id = "industry_银行" 等，node_type="industry"
- macro 节点：node_id = "macro_fx_usdcny" 等，node_type="macro"

边约定（有向）：
- belongs_to : stock -> industry（行业归属）
- correlates : stock -> stock（关联关系，如供应链/同概念）
- affects    : macro -> stock（宏观指标影响个股）

未来接入 GNN（DGL/PyG）时，GraphStore 的存储与查询接口保持不变。
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from data.external_store import load_graph

logger = logging.getLogger(__name__)

NODE_STOCK = "stock"
NODE_INDUSTRY = "industry"
NODE_MACRO = "macro"

EDGE_BELONGS_TO = "belongs_to"
EDGE_CORRELATES = "correlates"
EDGE_AFFECTS = "affects"


def stock_node_id(code: str) -> str:
    """股票代码 → 节点 ID（约定股票节点直接用代码作为 node_id）。"""
    return code


class GraphStore:
    """图查询与特征聚合。"""

    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame):
        self.nodes = nodes
        self.edges = edges

    @classmethod
    def from_disk(cls) -> "GraphStore":
        nodes, edges = load_graph()
        return cls(nodes, edges)

    @property
    def is_empty(self) -> bool:
        return self.nodes.empty or self.edges.empty

    def node_ids(self, node_type: Optional[str] = None) -> List[str]:
        nodes = self.nodes
        if node_type is not None:
            nodes = nodes[nodes["node_type"] == node_type]
        return nodes["node_id"].tolist()

    def neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """返回一阶邻居节点 ID（含出边与入边，去重保序）。"""
        edges = self.edges
        if edge_type is not None:
            edges = edges[edges["edge_type"] == edge_type]
        out = edges[edges["src"] == node_id]["dst"].tolist()
        out += edges[edges["dst"] == node_id]["src"].tolist()
        return list(dict.fromkeys(out))

    def neighbor_info(self, node_id: str, edge_type: Optional[str] = None) -> pd.DataFrame:
        """返回邻居节点信息（node_id/node_type/code/name/edge_type）。"""
        edges = self.edges
        if edge_type is not None:
            edges = edges[edges["edge_type"] == edge_type]

        out_rows = edges[edges["src"] == node_id][["dst", "edge_type"]].rename(columns={"dst": "node_id"})
        in_rows = edges[edges["dst"] == node_id][["src", "edge_type"]].rename(columns={"src": "node_id"})
        nb = pd.concat([out_rows, in_rows], ignore_index=True).drop_duplicates(subset=["node_id"])
        if nb.empty:
            return pd.DataFrame(columns=["node_id", "edge_type", "node_type", "code", "name"])
        return nb.merge(self.nodes, on="node_id", how="left")

    def aggregate(self, node_id: str, node_features: pd.DataFrame) -> pd.Series:
        """聚合一阶邻居特征，返回图特征 Series。

        node_features : DataFrame，index = node_id，列为数值特征。
        对每个 edge_type 与每个数值列，取该类型邻居的均值，
        生成列名 g_{edge_type}_{col}（无邻居/无特征时为 NaN）。
        """
        out = {}
        for etype in sorted(self.edges["edge_type"].unique()):
            nb_ids = self.neighbors(node_id, edge_type=etype)
            if not nb_ids:
                continue
            feats = node_features.reindex(nb_ids).dropna(how="all")
            if feats.empty:
                continue
            for col in feats.columns:
                out[f"g_{etype}_{col}"] = float(feats[col].mean())
        return pd.Series(out, dtype="float64")
