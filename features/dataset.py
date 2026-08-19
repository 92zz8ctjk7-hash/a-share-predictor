"""PyTorch Dataset 与数据集切分。

切分必须在时间维度进行（训练在前、验证其次、测试最后），
不能随机打乱，否则会引入未来信息泄漏。支持按全局日期排序后切分，
以及逐股票（panel）切分两种模式。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

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
                  valid_mask 非 None 时仅保留同股窗口（多股联合训练必选）
    """

    def __init__(
        self,
        features: np.ndarray,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
        seq_len: int = 10,
        mode: str = "flat",
        valid_mask: Optional[np.ndarray] = None,
    ):
        self.features = features
        self.y_reg = y_reg
        self.y_cls = y_cls
        self.seq_len = seq_len
        self.mode = mode

        if mode == "seq":
            total = len(features) - seq_len + 1
            if valid_mask is not None:
                self.idx = np.where(valid_mask[:total])[0]
            else:
                self.idx = np.arange(max(0, total))
            self.n_samples = len(self.idx)
        else:
            self.n_samples = len(features)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        if self.mode == "seq":
            i = int(self.idx[idx])
            x = self.features[i: i + self.seq_len]
            y_reg = self.y_reg[i + self.seq_len - 1]
            y_cls = self.y_cls[i + self.seq_len - 1]
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


# ---- 特征标准化 scaler 持久化（训练/推理对齐）----

SCALER_PATH = None  # 延迟解析，避免循环导入 config


def _scaler_path():
    from config import DATA_DIR

    return DATA_DIR / "meta" / "feature_scaler.parquet"


def _save_feature_scaler(feature_names, mu, sd) -> None:
    """保存训练段拟合的标准化参数，供独立推理路径对齐。"""
    import pandas as pd

    p = _scaler_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"feature": feature_names, "mean": mu, "std": sd}
    ).to_parquet(p, index=False)
    logger.info("特征标准化参数已保存：%s（%d 列）", p, len(feature_names))


def load_feature_scaler():
    """读取标准化参数（DataFrame，index=feature）；不存在返回 None。"""
    import pandas as pd

    p = _scaler_path()
    if not p.exists():
        return None
    return pd.read_parquet(p).set_index("feature")


def apply_feature_scaler(df: "pd.DataFrame", feat_cols: list) -> list:
    """对 df 的 feat_cols 应用已保存的标准化（按列名对齐）。

    返回实际被标准化的列（scaler 缺失的列保持原值并告警）。
    """
    sc = load_feature_scaler()
    if sc is None:
        logger.warning("特征标准化参数不存在，推理使用原始量级（与训练口径不一致）")
        return []
    hit = [c for c in feat_cols if c in sc.index]
    miss = [c for c in feat_cols if c not in sc.index]
    if miss:
        logger.warning("以下特征无标准化参数，保持原值：%s", miss)
    if hit:
        mu = sc.loc[hit, "mean"].to_numpy(dtype=np.float64)
        sd = sc.loc[hit, "std"].to_numpy(dtype=np.float64)
        df[hit] = (df[hit].to_numpy(dtype=np.float64) - mu) / sd
    return hit


def _seq_valid_mask(df: pd.DataFrame, seq_len: int) -> Optional[np.ndarray]:
    """seq 模式下每个窗口起点的合法性：窗口内不含股票切换。

    返回与滑窗样本等长的 bool 数组；单股或无 code 列时返回 None（全部合法）。
    """
    if "code" not in df.columns or len(df) < seq_len:
        return None
    codes = df["code"].to_numpy()
    boundary = np.zeros(len(codes), dtype=bool)
    boundary[1:] = codes[1:] != codes[:-1]
    if not boundary.any():
        return None
    total = len(codes) - seq_len + 1
    # 窗口 [i, i+seq_len-1] 内含切换点 → 非法；
    # 等价于起点 i 落在某股末尾 seq_len-1 个位置之内
    mask = np.ones(total, dtype=bool)
    b_idx = np.where(boundary)[0]
    for b in b_idx:
        lo, hi = max(0, b - seq_len + 1), min(total, b)
        mask[lo:hi] = False
    return mask


def _build_one(
    df: pd.DataFrame,
    seq_len: int,
    mode: str,
    feature_names: list,
) -> StockDataset:
    X = df[feature_names].to_numpy(dtype=np.float32)
    y_reg = df["label_reg"].to_numpy(dtype=np.float32)
    y_cls = df["label_cls"].to_numpy(dtype=np.int64)
    mask = _seq_valid_mask(df, seq_len) if mode == "seq" else None
    return StockDataset(X, y_reg, y_cls, seq_len=seq_len, mode=mode, valid_mask=mask)


def build_bundle(
    sample_df: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    seq_len: int = 10,
    mode: str = "flat",  # flat | seq
    persist_scaler: bool = False,
) -> DataBundle:
    """按全局时间顺序切分样本表。

    先按 code 分组、组内按 date 排序，再按「code」顺序拼接
    （seq 模式的滑窗必须落在同一股票内，跨股窗口会被剔除），
    最后在整条序列上做训练/验证/测试切分。
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

    # 多股联合训练时必须按 code 优先排序：同一股票的样本连续排列，
    # seq 模式的滑窗才不会跨股票拼接（date 优先排序会把同日不同股
    # 的行相邻放置，导致序列窗口 100% 跨股、模型退化为常数预测）
    ordered = (
        sample_df.sort_values(["code", "date"])
        .reset_index(drop=True)
    )

    n = len(ordered)
    if "date" in ordered.columns:
        # 严格按全局时间切分（防泄漏）：切点取日期的分位数，
        # 所有股票按同一切点划分 train/valid/test
        dates_sorted = ordered["date"].sort_values()
        train_end_date = dates_sorted.iloc[int(n * train_ratio) - 1]
        valid_end_date = dates_sorted.iloc[int(n * (train_ratio + valid_ratio)) - 1]
        train_df = ordered[ordered["date"] <= train_end_date].reset_index(drop=True)
        valid_df = ordered[
            (ordered["date"] > train_end_date) & (ordered["date"] <= valid_end_date)
        ].reset_index(drop=True)
        test_df = ordered[ordered["date"] > valid_end_date].reset_index(drop=True)
    else:
        train_end, valid_end = _split_mask(n, train_ratio, valid_ratio)
        train_df = ordered.iloc[:train_end].reset_index(drop=True)
        valid_df = ordered.iloc[train_end:valid_end].reset_index(drop=True)
        test_df = ordered.iloc[valid_end:].reset_index(drop=True)

    # 有效特征列：以样本表实际存在的列为准（如复权模式下无 turn），
    # 并包含外部注入的特征列（宏观/图特征等）
    feature_names = _feature_cols_of(sample_df)
    # 分片合并后，个别分片独有的列（如指数的 turn）在其他分片全为 NaN，
    # 稀疏列会污染训练输入导致 loss=nan，这里按 NaN 占比剔除
    nan_ratio = ordered[feature_names].isna().mean()
    dropped = nan_ratio[nan_ratio > 0.5].index.tolist()
    if dropped:
        logger.warning("剔除稀疏特征列（NaN>50%%）：%s", dropped)
        feature_names = [c for c in feature_names if c not in dropped]
    
    # 特征标准化（z-score，仅用训练段统计量防泄漏）：
    # 原始特征量级跨 6 个数量级（货币供应 ~3e4 vs 日收益 ~1e-2），
    # 不标准化时 LSTM 梯度被大量级慢变特征主导，退化为常数预测器
    mu = train_df[feature_names].mean().to_numpy(dtype=np.float64)
    sd = train_df[feature_names].std().to_numpy(dtype=np.float64)
    sd = np.where(sd < 1e-8, 1.0, sd)
    for _df in (train_df, valid_df, test_df):
        _df[feature_names] = (
            _df[feature_names].to_numpy(dtype=np.float64) - mu
        ) / sd
    if persist_scaler:
        _save_feature_scaler(feature_names, mu, sd)
    
    train_set = _build_one(train_df, seq_len, mode, feature_names)
    valid_set = _build_one(valid_df, seq_len, mode, feature_names)
    test_set = _build_one(test_df, seq_len, mode, feature_names)

    # seq 模式下，Dataset 仅保留同股窗口（每只股票前 seq_len-1 个位置 +
    # 跨股衔接处无完整回溯窗口），同步截断存储的 df 保证标签与预测对齐。
    if mode == "seq":
        def _trim_seq(df: pd.DataFrame) -> pd.DataFrame:
            mask = _seq_valid_mask(df, seq_len)
            if mask is None:
                if "code" in df.columns:
                    m = df.groupby("code").cumcount() >= seq_len - 1
                    return df[m].reset_index(drop=True)
                return df.iloc[seq_len - 1:].reset_index(drop=True)
            total = len(df) - seq_len + 1
            # 窗口的目标行 = 起点 + seq_len - 1，仅保留合法同股窗口的目标行
            tgt = np.zeros(len(df), dtype=bool)
            tgt[(seq_len - 1):(seq_len - 1 + total)] = mask
            return df[tgt].reset_index(drop=True)

        train_df = _trim_seq(train_df)
        valid_df = _trim_seq(valid_df)
        test_df = _trim_seq(test_df)

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
