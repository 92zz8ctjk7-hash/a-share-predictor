"""分钟级样本的 PyTorch Dataset 与时间切分。

与日线 dataset.py 相同，切分严格按全局日期顺序（训练在前、验证其次、
测试最后），不可随机打乱，避免未来信息泄漏。
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # 未安装 torch 时仅可做数据切分
    torch = None
    Dataset = object

from features.min_builder import MIN_SEQ_COLUMNS, SEQ_FEATURES, SEQ_LEN

logger = logging.getLogger(__name__)

# 元信息与标签列
_META_COLS = ("date", "code", "label_rest", "label_cls_rest")


class MinStockDataset(Dataset):
    """分钟级样本 Dataset：返回 (x_seq, x_static, y_reg, y_cls)。

    x_seq   : (SEQ_LEN, len(SEQ_FEATURES))，开盘 30 分钟序列
    x_static: (static_dim,)，日线特征 + 压力位 + 基座预测
    """

    def __init__(
        self,
        seq: np.ndarray,
        static: np.ndarray,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
    ):
        self.seq = seq
        self.static = static
        self.y_reg = y_reg
        self.y_cls = y_cls

    def __len__(self) -> int:
        return len(self.y_reg)

    def __getitem__(self, idx: int):
        x_seq = torch.tensor(self.seq[idx], dtype=torch.float32)
        x_static = torch.tensor(self.static[idx], dtype=torch.float32)
        y_reg = torch.tensor([self.y_reg[idx]], dtype=torch.float32)
        y_cls = torch.tensor([self.y_cls[idx]], dtype=torch.long)
        return x_seq, x_static, y_reg, y_cls


class MinDataBundle:
    """包含 train/valid/test 三组 MinStockDataset。"""

    def __init__(
        self,
        train: MinStockDataset,
        valid: MinStockDataset,
        test: MinStockDataset,
        seq_input_dim: int,
        static_cols: List[str],
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        self.train = train
        self.valid = valid
        self.test = test
        self.seq_input_dim = seq_input_dim
        self.static_cols = static_cols
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df


def _static_cols_of(df: pd.DataFrame) -> List[str]:
    """样本表中的静态特征列 = 除元信息与序列列之外的列。"""
    return [
        c for c in df.columns
        if c not in _META_COLS and not c.startswith("seq_")
    ]


def _extract(df: pd.DataFrame, static_cols: List[str]):
    seq = (
        df[MIN_SEQ_COLUMNS].to_numpy(dtype=np.float32)
        .reshape(-1, SEQ_LEN, len(SEQ_FEATURES))
    )
    static = df[static_cols].to_numpy(dtype=np.float32)
    y_reg = df["label_rest"].to_numpy(dtype=np.float32)
    y_cls = df["label_cls_rest"].to_numpy(dtype=np.int64)
    return seq, static, y_reg, y_cls


def build_min_bundle(
    sample_df: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> MinDataBundle:
    """按全局日期顺序切分分钟样本表。

    参数：
        sample_df   : load_all_min_samples 返回的合并长表
        train_ratio : 训练占比
        valid_ratio : 验证占比（其余为测试集）
    """
    if sample_df.empty:
        raise ValueError("分钟样本表为空，无法切分")

    static_cols = _static_cols_of(sample_df)
    ordered = sample_df.sort_values(["date", "code"]).reset_index(drop=True)

    n = len(ordered)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_df = ordered.iloc[:train_end].reset_index(drop=True)
    valid_df = ordered.iloc[train_end:valid_end].reset_index(drop=True)
    test_df = ordered.iloc[valid_end:].reset_index(drop=True)

    def build_one(df: pd.DataFrame) -> MinStockDataset:
        seq, static, y_reg, y_cls = _extract(df, static_cols)
        return MinStockDataset(seq, static, y_reg, y_cls)

    logger.info(
        "分钟样本切分完成：train=%d valid=%d test=%d",
        len(train_df), len(valid_df), len(test_df),
    )
    return MinDataBundle(
        train=build_one(train_df),
        valid=build_one(valid_df),
        test=build_one(test_df),
        seq_input_dim=len(SEQ_FEATURES),
        static_cols=static_cols,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )
