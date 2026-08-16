"""特征工程与标签构造。

计算技术指标作为特征，并构造「未来 N 日涨跌幅」与「涨跌方向」双标签。

关键点：标签使用未来数据计算，因此构建样本时约定：
- 特征只用截至 t 时刻（含 t 当日收盘）的信息
- 标签用 t+1 .. t+N 的数据，N 为 horizon
这样可保证在实际交易中特征与标签在时间上不重叠（无未来函数泄漏）。

切分数据集时必须按时间顺序（见 dataset.py），不可随机打乱。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data import store

logger = logging.getLogger(__name__)


def compute_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """在单只股票 DataFrame 上计算技术特征。

    df 必须已按 date 升序排列，含 close/high/low/volume/amount/turn/pct_chg。
    返回新增特征列后的 DataFrame。
    """
    df = df.copy()
    close = df["close"]

    # 收益率
    df["ret_1d"] = close.pct_change()

    # 移动均线
    for w in [5, 10, 20]:
        if len(df) >= w:
            df[f"ma_{w}"] = close.rolling(w).mean()
        else:
            df[f"ma_{w}"] = np.nan

    # 均线偏离度
    df["ma_bias_5"] = close / df["ma_5"] - 1
    df["ma_bias_20"] = close / df["ma_20"] - 1

    # 价格相对位置（过去 window 日内的位置 0~1）
    df["roll_high"] = df["high"].rolling(window).max()
    df["roll_low"] = df["low"].rolling(window).min()
    df["price_position"] = (close - df["roll_low"]) / (
        df["roll_high"] - df["roll_low"] + 1e-9
    )

    # 波动率（rolling std）
    df["volatility"] = df["ret_1d"].rolling(window).std()

    # 量比（当日成交量 / 过去 window 日均量）
    df["volume_ratio"] = df["volume"] / (
        df["volume"].rolling(window).mean() + 1e-9
    )

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD(12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    df["macd"] = (dif - dea) * 2
    df["macd_dif"] = dif
    df["macd_dea"] = dea

    # 振幅
    df["amplitude"] = (df["high"] - df["low"]) / (
        df["preclose"].replace(0, np.nan) + 1e-9
    )
    return df


def compute_sr_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算压力位/支撑位相关特征（基于日线，单只股票）。

    新增列（均以 % 表示，正值=现价高于该位置）：
        dist_high20 / dist_high60 : 收盘价距 20/60 日滚动高点的距离
        dist_low20  / dist_low60  : 收盘价距 20/60 日滚动低点的距离
        boll_up_bias / boll_low_bias : 收盘价相对布林带(20, 2σ)上下轨位置

    独立于 compute_features，不影响现有日线训练的特征列。
    """
    df = df.copy()
    close = df["close"]

    for w in (20, 60):
        roll_high = close.rolling(w).max()
        roll_low = close.rolling(w).min()
        df[f"dist_high{w}"] = (close / roll_high - 1) * 100.0
        df[f"dist_low{w}"] = (close / roll_low - 1) * 100.0

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    boll_up = ma20 + 2 * std20
    boll_low = ma20 - 2 * std20
    df["boll_up_bias"] = (close - boll_up) / close * 100.0
    df["boll_low_bias"] = (close - boll_low) / close * 100.0
    return df


# 压力位/支撑位特征列（分钟级模型静态输入用）
SR_FEATURE_COLUMNS: List[str] = [
    "dist_high20", "dist_high60", "dist_low20", "dist_low60",
    "boll_up_bias", "boll_low_bias",
]


def add_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """构造双标签。

    label_reg : 未来第 horizon 个交易日相对当日收盘的涨跌幅（%）
    label_cls : 1 = 上涨，0 = 下跌（与 label_reg > 0 一致）
    """
    df = df.copy()
    future_close = df["close"].shift(-horizon)
    df["label_reg"] = (future_close / df["close"] - 1) * 100.0
    df["label_cls"] = (df["label_reg"] > 0).astype(int)
    return df


# 用于模型的特征列（排除原始价量中的冗余与标签列）
FEATURE_COLUMNS: List[str] = [
    "ret_1d",
    "ma_5",
    "ma_10",
    "ma_20",
    "ma_bias_5",
    "ma_bias_20",
    "price_position",
    "volatility",
    "volume_ratio",
    "rsi_14",
    "macd",
    "macd_dif",
    "macd_dea",
    "amplitude",
    "turn",
]

# 标签列
LABEL_COLUMNS: List[str] = ["label_reg", "label_cls"]


def build_dataset(
    raw: pd.DataFrame,
    window: int = 20,
    horizon: int = 5,
) -> pd.DataFrame:
    """将多只股票的长表数据构建为带特征与标签的样本表。

    参数：
        raw    : fetch_many 返回的长表（含 code/date/...）
        window : 回看窗口
        horizon: 预测周期

    返回：
        含特征列与标签列的样本 DataFrame，仅保留标签非 NaN 的行。
    """
    if raw.empty:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for code, g in raw.groupby("code"):
        g = g.sort_values("date").reset_index(drop=True)
        g = compute_features(g, window=window)
        g = add_labels(g, horizon=horizon)
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)

    meta_cols = ["date", "code", "close", "label_reg", "label_cls"]
    keep = meta_cols + FEATURE_COLUMNS
    keep = [c for c in keep if c in out.columns]

    out = out[keep]
    # 复权模式下 turn 全为 NaN（baostock 不返回换手率），直接剔除该列；
    # 其余 NaN 来自特征窗口边界与未来标签缺失，整行删除
    all_nan = [
        c for c in FEATURE_COLUMNS
        if c in out.columns and out[c].isna().all()
    ]
    if all_nan:
        out = out.drop(columns=all_nan)
    valid_features = [c for c in FEATURE_COLUMNS if c in out.columns]
    out = out.dropna(subset=valid_features + ["label_reg"]).reset_index(drop=True)
    return out


def build_samples_to_store(
    window: int = 20,
    horizon: int = 5,
    overwrite: bool = False,
    codes: Optional[List[str]] = None,
) -> List[str]:
    """从 raw 分片目录流式构建 samples 分片（每只股票一个文件）。

    遍历 data.store.RAW_DIR 下所有分片，逐只计算特征与标签后
    写入 data.store.SAMPLES_DIR。内存占用恒定，适合全市场数据；
    中断后重跑会自动跳过已构建的股票（overwrite=False 时）。

    参数：
        window   : 回看窗口
        horizon  : 预测周期
        overwrite: 是否重建已存在的 samples 分片
        codes    : 仅构建指定股票（None = raw 目录中的全部）

    返回：本次成功构建的股票代码列表。
    """
    raw_codes = store.list_raw_codes()
    if codes is not None:
        raw_codes = [c for c in codes if c in raw_codes]
    if not raw_codes:
        logger.warning("raw 分片目录为空，请先运行 fetch --all")
        return []

    # 外部数据自动检测（无数据时行为与之前完全一致）
    from features.external import load_graph_features, load_macro_features
    from features.market import load_market_features

    macro_feats = load_macro_features()  # index=交易日，或 None
    market_feats = load_market_features()  # 指数环境+日历，或 None
    graph_feats = load_graph_features()  # {code: 图特征 Series}，或 {}
    if macro_feats is not None or graph_feats or market_feats is not None:
        logger.info(
            "检测到外部数据：macro=%s market=%s graph=%d 只，将注入基座样本",
            "yes" if macro_feats is not None else "no",
            "yes" if market_feats is not None else "no",
            len(graph_feats),
        )

    built: List[str] = []
    skipped = 0
    for i, code in enumerate(raw_codes):
        if store.sample_path(code).exists() and not overwrite:
            skipped += 1
            continue

        df = store.load_bars(code)
        if df is None or df.empty:
            logger.warning("raw 分片为空，跳过 %s", code)
            continue

        sample = build_dataset(df, window=window, horizon=horizon)
        if sample.empty:
            logger.warning("样本为空（数据不足），跳过 %s", code)
            continue

        # 注入宏观特征（基座为收盘后视角，按样本当日 join）
        if macro_feats is not None:
            n_before = len(sample.columns)
            sample = sample.merge(macro_feats.reset_index(), on="date", how="left")
            new_cols = sample.columns[n_before:]
            sample[new_cols] = sample[new_cols].fillna(0.0)

        # 注入市场环境特征（指数环境+日历效应，按样本当日 join）
        if market_feats is not None:
            n_before = len(sample.columns)
            sample = sample.merge(market_feats.reset_index(), on="date", how="left")
            new_cols = sample.columns[n_before:]
            sample[new_cols] = sample[new_cols].fillna(0.0)

        # 注入图特征（按股票静态注入）
        if code in graph_feats:
            for k, v in graph_feats[code].items():
                sample[k] = v

        store.save_samples(sample, code)
        built.append(code)
        if (i + 1) % 500 == 0:
            logger.info(
                "特征构建进度 %d/%d（已构建 %d，跳过 %d）",
                i + 1, len(raw_codes), len(built), skipped,
            )

    logger.info(
        "samples 分片构建完成：构建 %d 只，跳过 %d 只，共 %d 只",
        len(built), skipped, len(raw_codes),
    )
    return built


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """抽取特征矩阵。"""
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[cols].to_numpy(dtype=np.float32)