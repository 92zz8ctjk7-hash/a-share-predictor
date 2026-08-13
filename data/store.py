"""本地数据存储层：定义全量数据的存储格式与读写 API。

目录结构（均在 cache/ 下）：

    cache/
    ├── meta/
    │   └── stock_list.parquet   # 全市场股票清单（baostock query_stock_basic 结果）
    ├── raw/                     # 原始日线，按股票分片（每只一个文件）
    │   ├── sh.600000.parquet
    │   └── sz.000001.parquet
    ├── raw_min/                 # 原始分钟线，按「频率/股票」两层分片
    │   ├── 5/                   # 5 分钟线（另有 15/30/60 分钟目录）
    │   │   ├── sh.600000.parquet
    │   │   └── sz.000001.parquet
    │   └── 15/
    ├── min_samples/             # 分钟级预测样本，按「频率/股票」两层分片
    │   ├── 5/
    │   │   ├── sh.600000.parquet
    │   │   └── sz.000001.parquet
    │   └── 15/
    └── samples/                 # 特征样本，按股票分片（每只一个文件）
        ├── sh.600000.parquet
        └── sz.000001.parquet

设计要点：
- raw 按 code 分片：天然支持断点续传（文件存在即跳过）、单只股票
  增量更新（按 date 去重合并）、流式特征构建（内存占用恒定）
- raw_min 按「频率/股票」分片：分钟线字段与日线不同（含 time，
  无 preclose/turn/pctChg），且只对部分股票拉取，单独目录隔离；
  仅保留最近约 1 年历史（baostock 限制）
- samples 按 code 分片：pandas/pyarrow 可直接按目录整体读取，
  等价于合并后的长表，无需维护单个大文件
- raw 数值列统一 float64（保留价量精度）；samples 特征列统一
  float32（技术指标为近似值，float32 可减半存储与内存）
- date 统一 datetime64[ns]，code 统一字符串

字段约定（与 data/schema.py 一致）：
    raw     : date code open high low close preclose volume amount turn pct_chg
    raw_min : date time code open high low close volume amount
    samples : date code close label_reg label_cls + FEATURE_COLUMNS
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)

# ---- 目录 ----
META_DIR = DATA_DIR / "meta"
RAW_DIR = DATA_DIR / "raw"
MIN_DIR = DATA_DIR / "raw_min"
MIN_SAMPLES_DIR = DATA_DIR / "min_samples"
SAMPLES_DIR = DATA_DIR / "samples"
for _d in (META_DIR, RAW_DIR, MIN_DIR, MIN_SAMPLES_DIR, SAMPLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# baostock 支持的分钟线频率
MIN_FREQUENCIES: List[str] = ["5", "15", "30", "60"]

# ---- Schema 规范 ----
# raw 层列顺序
RAW_COLUMNS: List[str] = [
    "date", "code", "open", "high", "low", "close",
    "preclose", "volume", "amount", "turn", "pct_chg",
]
# raw 层 dtype 规范
RAW_DTYPES = {
    "date": "datetime64[ns]",
    "code": "string",
    **{c: "float64" for c in RAW_COLUMNS[2:]},
}

# samples 层：标签与元信息列（特征列来自 features.builder.FEATURE_COLUMNS）
SAMPLE_META_COLUMNS: List[str] = ["date", "code", "close", "label_reg", "label_cls"]
# samples 层 dtype 规范（特征列统一 float32）
SAMPLE_DTYPES = {
    "date": "datetime64[ns]",
    "code": "string",
    "close": "float32",
    "label_reg": "float32",
    "label_cls": "int8",
}


def raw_path(code: str) -> Path:
    """raw 分片文件路径，如 sh.600000 -> cache/raw/sh.600000.parquet"""
    safe = code.replace(".", "_")
    return RAW_DIR / f"{safe}.parquet"


def sample_path(code: str) -> Path:
    """samples 分片文件路径，如 sh.600000 -> cache/samples/sh.600000.parquet"""
    safe = code.replace(".", "_")
    return SAMPLES_DIR / f"{safe}.parquet"


def list_raw_codes() -> List[str]:
    """列出已缓存 raw 分片的股票代码（按文件名解析，不读内容）。"""
    codes = []
    for f in sorted(RAW_DIR.glob("*.parquet")):
        codes.append(f.stem.replace("_", "."))
    return codes


def list_sample_codes() -> List[str]:
    """列出已缓存 samples 分片的股票代码。"""
    codes = []
    for f in sorted(SAMPLES_DIR.glob("*.parquet")):
        codes.append(f.stem.replace("_", "."))
    return codes


# ---- raw 层读写 ----

def load_bars(code: str) -> Optional[pd.DataFrame]:
    """读取单只股票的 raw 分片；不存在返回 None。"""
    path = raw_path(code)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_bars(df: pd.DataFrame, code: str, merge: bool = False) -> None:
    """保存单只股票的 raw 分片。

    merge=True 时与已有分片按 date 去重合并（用于增量更新）；
    merge=False 时直接覆盖写。
    """
    out = _coerce_raw(df, code)
    path = raw_path(code)
    if merge and path.exists():
        old = pd.read_parquet(path)
        out = (
            pd.concat([old, out], ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
    out.to_parquet(path, index=False)


def _coerce_raw(df: pd.DataFrame, code: Optional[str] = None) -> pd.DataFrame:
    """校验并规范化 raw DataFrame：列顺序、dtype、排序。"""
    out = df.copy()
    if code is not None:
        out["code"] = code
    missing = [c for c in RAW_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"raw 数据缺少字段: {missing}")
    out = out[RAW_COLUMNS]
    for col, dtype in RAW_DTYPES.items():
        out[col] = out[col].astype(dtype)
    return out.sort_values("date").reset_index(drop=True)


# ---- samples 层读写 ----

def save_samples(df: pd.DataFrame, code: str) -> None:
    """保存单只股票的 samples 分片（覆盖写，特征列转 float32）。"""
    out = _coerce_samples(df, code)
    out.to_parquet(sample_path(code), index=False)


def _coerce_samples(df: pd.DataFrame, code: Optional[str] = None) -> pd.DataFrame:
    """校验并规范化 samples DataFrame：列顺序、dtype、排序。

    列 = SAMPLE_META_COLUMNS + FEATURE_COLUMNS + 外部注入特征列
    （外部特征如宏观/图特征，按存在即保留处理）。
    """
    from features.builder import FEATURE_COLUMNS

    out = df.copy()
    if code is not None:
        out["code"] = code

    keep = SAMPLE_META_COLUMNS + [c for c in FEATURE_COLUMNS if c in out.columns]
    missing = [c for c in keep if c not in out.columns]
    if missing:
        raise ValueError(f"samples 数据缺少字段: {missing}")
    extra = [c for c in out.columns if c not in keep]
    out = out[keep + extra]

    for col, dtype in SAMPLE_DTYPES.items():
        out[col] = out[col].astype(dtype)
    for col in FEATURE_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("float32")
    for col in extra:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out.sort_values("date").reset_index(drop=True)


def load_all_samples() -> pd.DataFrame:
    """读取全部 samples 分片，返回合并长表。

    pandas/pyarrow 直接扫描 SAMPLES_DIR 下所有 parquet 文件，
    等价于「所有分片按行拼接」，无需维护合并大文件。
    """
    files = sorted(SAMPLES_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"samples 目录为空: {SAMPLES_DIR}")
    df = pd.read_parquet(SAMPLES_DIR)
    logger.info("读取样本分片 %d 个，共 %d 行", len(files), len(df))
    return df


def load_all_raw() -> pd.DataFrame:
    """读取全部 raw 分片，返回合并长表（仅用于小批量/调试场景）。"""
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"raw 目录为空: {RAW_DIR}")
    df = pd.read_parquet(RAW_DIR)
    logger.info("读取 raw 分片 %d 个，共 %d 行", len(files), len(df))
    return df


# ---- raw_min（分钟线）层读写 ----

# 分钟线列顺序
MIN_COLUMNS: List[str] = [
    "date", "time", "code", "open", "high", "low", "close",
    "volume", "amount",
]
# 分钟线 dtype 规范
MIN_DTYPES = {
    "date": "datetime64[ns]",
    "time": "datetime64[ns]",
    "code": "string",
    **{c: "float64" for c in MIN_COLUMNS[3:]},
}


def min_dir(frequency: str) -> Path:
    """分钟线分片目录，如 frequency="5" -> cache/raw_min/5"""
    if frequency not in MIN_FREQUENCIES:
        raise ValueError(f"不支持的分钟线频率: {frequency}，可选 {MIN_FREQUENCIES}")
    d = MIN_DIR / frequency
    d.mkdir(parents=True, exist_ok=True)
    return d


def min_path(code: str, frequency: str) -> Path:
    """分钟线分片文件路径，如 (sh.600000, "5") -> cache/raw_min/5/sh.600000.parquet"""
    safe = code.replace(".", "_")
    return min_dir(frequency) / f"{safe}.parquet"


def list_min_codes(frequency: str) -> List[str]:
    """列出某频率下已缓存的分钟线股票代码。"""
    codes = []
    for f in sorted(min_dir(frequency).glob("*.parquet")):
        codes.append(f.stem.replace("_", "."))
    return codes


def load_min_bars(code: str, frequency: str) -> Optional[pd.DataFrame]:
    """读取单只股票的分钟线分片；不存在返回 None。"""
    path = min_path(code, frequency)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_min_bars(df: pd.DataFrame, code: str, frequency: str, merge: bool = False) -> None:
    """保存单只股票的分钟线分片。

    merge=True 时与已有分片按 (date, time) 去重合并（用于增量更新）；
    merge=False 时直接覆盖写。
    """
    out = _coerce_min(df, code)
    path = min_path(code, frequency)
    if merge and path.exists():
        old = pd.read_parquet(path)
        out = (
            pd.concat([old, out], ignore_index=True)
            .drop_duplicates(subset=["date", "time"], keep="last")
            .sort_values(["date", "time"])
            .reset_index(drop=True)
        )
    out.to_parquet(path, index=False)


def _coerce_min(df: pd.DataFrame, code: Optional[str] = None) -> pd.DataFrame:
    """校验并规范化分钟线 DataFrame：列顺序、dtype、排序。"""
    out = df.copy()
    if code is not None:
        out["code"] = code
    missing = [c for c in MIN_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"分钟线数据缺少字段: {missing}")
    out = out[MIN_COLUMNS]
    for col, dtype in MIN_DTYPES.items():
        out[col] = out[col].astype(dtype)
    return out.sort_values(["date", "time"]).reset_index(drop=True)


# ---- min_samples（分钟级预测样本）层读写 ----

# 样本元信息与标签列（序列/静态特征列见 features.min_builder）
MIN_SAMPLE_META_COLUMNS: List[str] = ["date", "code", "label_rest", "label_cls_rest"]
MIN_SAMPLE_DTYPES = {
    "date": "datetime64[ns]",
    "code": "string",
    "label_rest": "float32",
    "label_cls_rest": "int8",
}


def min_samples_dir(frequency: str) -> Path:
    """分钟样本分片目录，如 frequency="5" -> cache/min_samples/5"""
    if frequency not in MIN_FREQUENCIES:
        raise ValueError(f"不支持的分钟线频率: {frequency}，可选 {MIN_FREQUENCIES}")
    d = MIN_SAMPLES_DIR / frequency
    d.mkdir(parents=True, exist_ok=True)
    return d


def min_sample_path(code: str, frequency: str) -> Path:
    """分钟样本分片文件路径，如 (sh.600000, "5") -> cache/min_samples/5/sh.600000.parquet"""
    safe = code.replace(".", "_")
    return min_samples_dir(frequency) / f"{safe}.parquet"


def list_min_sample_codes(frequency: str) -> List[str]:
    """列出某频率下已构建分钟样本的股票代码。"""
    codes = []
    for f in sorted(min_samples_dir(frequency).glob("*.parquet")):
        codes.append(f.stem.replace("_", "."))
    return codes


def save_min_samples(df: pd.DataFrame, code: str, frequency: str) -> None:
    """保存单只股票的分钟样本分片（覆盖写，数值列转 float32）。"""
    out = _coerce_min_samples(df, code)
    out.to_parquet(min_sample_path(code, frequency), index=False)


def load_min_samples(code: str, frequency: str) -> Optional[pd.DataFrame]:
    """读取单只股票的分钟样本分片；不存在返回 None。"""
    path = min_sample_path(code, frequency)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_all_min_samples(frequency: str) -> pd.DataFrame:
    """读取某频率下全部分钟样本分片，返回合并长表。"""
    files = sorted(min_samples_dir(frequency).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"min_samples 目录为空: {min_samples_dir(frequency)}")
    df = pd.read_parquet(min_samples_dir(frequency))
    logger.info("读取分钟样本分片 %d 个，共 %d 行", len(files), len(df))
    return df


def _coerce_min_samples(df: pd.DataFrame, code: Optional[str] = None) -> pd.DataFrame:
    """校验并规范化分钟样本 DataFrame：列顺序、dtype、排序。

    列 = MIN_SAMPLE_META_COLUMNS + 序列特征列 + 静态特征列
    （序列/静态特征列名来自 features.min_builder，延迟导入避免循环依赖）。
    """
    from features.min_builder import MIN_SEQ_COLUMNS, MIN_STATIC_COLUMNS

    out = df.copy()
    if code is not None:
        out["code"] = code

    # 序列列必须完整；静态列按实际存在保留（turn 等在复权模式下被剔除），
    # 外部注入特征列（基本面/宏观等）按存在即保留
    missing = [c for c in MIN_SAMPLE_META_COLUMNS + MIN_SEQ_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"分钟样本缺少字段: {missing}")
    static_present = [c for c in MIN_STATIC_COLUMNS if c in out.columns]
    extra = [c for c in out.columns if c not in MIN_SAMPLE_META_COLUMNS + MIN_SEQ_COLUMNS + static_present]
    out = out[MIN_SAMPLE_META_COLUMNS + MIN_SEQ_COLUMNS + static_present + extra]

    for col, dtype in MIN_SAMPLE_DTYPES.items():
        out[col] = out[col].astype(dtype)
    for col in MIN_SEQ_COLUMNS + static_present + extra:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out.sort_values("date").reset_index(drop=True)


# ---- 股票清单 ----

def save_stock_list(df: pd.DataFrame) -> None:
    """保存全市场股票清单（query_stock_basic 原始结果）。"""
    df.to_parquet(META_DIR / "stock_list.parquet", index=False)


def load_stock_list() -> Optional[pd.DataFrame]:
    """读取股票清单；不存在返回 None。"""
    path = META_DIR / "stock_list.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)
