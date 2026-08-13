"""PyTorch Dataset 与数据集切分。

切分必须在时间维度进行（训练在前、验证其次、测试最后），
不能随机打乱，否则会引入未来信息泄漏。支持按全局日期排序后切分，
以及逐股票（panel）切分两种模式。
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # 未安装 torch 时，仅支持 sklearn 基线的数据切分
    torch = None
    Dataset = object

from features.builder import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class StockDataset(Dataset):
    """支持 MLP 与 LSTM 的通用 Dataset。

    mode="flat" : 返回 (x, y_reg, y_cls)，x 形状 (n_features,)
    mode="seq"  : 返回 (x, y_reg, y_cls)，x 形状 (seq_len, n_features)
    """

    def __init__(
        self,
        features: np.ndarray,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
        seq_len: int = 10,
        mode: str = "flat",
    ):
        self.features = features
        self.y_reg = y_reg
        self.y_cls = y_cls
        self.seq_len = seq_len
        self.mode = mode

        if mode == "seq":
            total = len(features) - seq_len + 1
            self.n_samples = max(0, total)
        else:
            self.n_samples = len(features)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        if self.mode == "seq":
            x = self.features[idx: idx + self.seq_len]
            y_reg = self.y_reg[idx + self.seq_len - 1]
            y_cls = self.y_cls[idx + self.seq_len - 1]
        else:
            x = self.features[idx]
            y_reg = self.y_reg[idx]
            y_cls = self.y_cls[idx]

        x_t = torch.tensor(x, dtype=torch.float32)
        y_reg_t = torch.tensor([y_reg], dtype=torch.float32)
        y_cls_t = torch.tensor([y_cls], dtype=torch.long)
        return x_t, y_reg_t, y_cls_t


# 样本表中的元信息与标签列
_META_COLS = ("date", "code", "close", "label_reg", "label_cls")


def _feature_cols_of(df: pd.DataFrame) -> list:
    """有效特征列 = 标准特征列 + 外部注入特征列（如宏观/图特征）。"""
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in _META_COLS and c not in cols]
    return cols + extra


class DataBundle:
    """包含 train/valid/test 三组数据，每一组都是 StockDataset。"""

    def __init__(
        self,
        train: StockDataset,
        valid: StockDataset,
        test: StockDataset,
        feature_names: list,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        self.train = train
        self.valid = valid
        self.test = test
        self.feature_names = feature_names
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

    def X_y(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回某个 split 的 (X, y_reg, y_cls)，供 sklearn 基线使用。"""
        df = getattr(self, f"{split}_df")
        X = df[self.feature_names].to_numpy(dtype=np.float32)
        y_reg = df["label_reg"].to_numpy(dtype=np.float32)
        y_cls = df["label_cls"].to_numpy(dtype=np.int64)
        return X, y_reg, y_cls


def _split_mask(length: int, train_ratio: float, valid_ratio: float):
    """生成时间顺序的切分索引。"""
    train_end = int(length * train_ratio)
    valid_end = int(length * (train_ratio + valid_ratio))
    return train_end, valid_end


def _build_one(
    df: pd.DataFrame,
    seq_len: int,
    mode: str,
    feature_names: list,
) -> StockDataset:
    X = df[feature_names].to_numpy(dtype=np.float32)
    y_reg = df["label_reg"].to_numpy(dtype=np.float32)
    y_cls = df["label_cls"].to_numpy(dtype=np.int64)
    return StockDataset(X, y_reg, y_cls, seq_len=seq_len, mode=mode)


def build_bundle(
    sample_df: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    seq_len: int = 10,
    mode: str = "flat",  # flat | seq
) -> DataBundle:
    """按全局时间顺序切分样本表。

    先按 code 分组、组内按 date 排序，再按「code」顺序拼接，
    最后在整条时间序列上做训练/验证/测试切分。
    这样不同股票的历史也遵循时间顺序，适合多标的联合训练。

    参数：
        sample_df   : build_dataset 返回的样本表（含 date/code/特征/标签）
        train_ratio : 训练占比
        valid_ratio : 验证占比
        seq_len     : LSTM 序列长度（mode="seq" 时生效）
        mode        : "flat"（MLP/sklearn）或 "seq"（LSTM）
    """
    if sample_df.empty:
        raise ValueError("样本表为空，无法切分")

    ordered = (
        sample_df.sort_values(["date", "code"])
        .reset_index(drop=True)
    )

    n = len(ordered)
    train_end, valid_end = _split_mask(n, train_ratio, valid_ratio)

    train_df = ordered.iloc[:train_end].reset_index(drop=True)
    valid_df = ordered.iloc[train_end:valid_end].reset_index(drop=True)
    test_df = ordered.iloc[valid_end:].reset_index(drop=True)

    # 有效特征列：以样本表实际存在的列为准（如复权模式下无 turn），
    # 并包含外部注入的特征列（宏观/图特征等）
    feature_names = _feature_cols_of(sample_df)
    train_set = _build_one(train_df, seq_len, mode, feature_names)
    valid_set = _build_one(valid_df, seq_len, mode, feature_names)
    test_set = _build_one(test_df, seq_len, mode, feature_names)

    # seq 模式下，Dataset 会丢弃前 seq_len-1 个样本（缺少足够的回溯窗口），
    # 因此同步截断存储的 df，保证后续评估时标签与预测一一对应。
    if mode == "seq":
        train_df = train_df.iloc[seq_len - 1:].reset_index(drop=True)
        valid_df = valid_df.iloc[seq_len - 1:].reset_index(drop=True)
        test_df = test_df.iloc[seq_len - 1:].reset_index(drop=True)

    logger.info(
        "数据切分完成：train=%d valid=%d test=%d (mode=%s)",
        len(train_set), len(valid_set), len(test_set), mode,
    )

    return DataBundle(
        train=train_set,
        valid=valid_set,
        test=test_set,
        feature_names=feature_names,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )
