"""分钟级样本构造：开盘 30 分钟截面 → 当日收盘涨跌预测样本。

对每只股票每个交易日 T 构造一条样本：
- 序列特征（LSTM 输入）：10:00 前 6 根 5 分钟线，每根 4 维
    ret      : 本根收盘 / 前根收盘 - 1（第一根相对前收）
    vs_open  : 本根收盘相对当日开盘价的涨跌
    vol_ratio: 本根成交量 / 前 5 日同时段均量
    amplitude: (high - low) / 当日开盘价
- 静态特征：T-1 日线技术特征（FEATURE_COLUMNS）+ 压力位特征
  （SR_FEATURE_COLUMNS）+ base_pred（基座模型对 T 日的预测，级联注入）
- 标签：label_rest = close(T)/close(10:00)-1（%）、label_cls_rest（0/1）

无泄漏约定：输入只使用 ≤10:00 的信息，标签只使用 15:00 收盘价。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data import store
from features.builder import (
    FEATURE_COLUMNS,
    SR_FEATURE_COLUMNS,
    compute_features,
    compute_sr_features,
)

logger = logging.getLogger(__name__)

# 10:00 截面：10:00 前（含）的 bar 数量（09:35/09:40/09:45/09:50/09:55/10:00）
SEQ_LEN = 6
CUTOFF_TIME = "10:00"

# 每根 bar 的 4 维序列特征
SEQ_FEATURES: List[str] = ["ret", "vs_open", "vol_ratio", "amplitude"]

# 展平后的序列特征列名（seq_0_ret ... seq_5_amplitude）
MIN_SEQ_COLUMNS: List[str] = [
    f"seq_{i}_{f}" for i in range(SEQ_LEN) for f in SEQ_FEATURES
]

# 静态特征列（turn 等在复权模式下可能缺失，由构造逻辑动态剔除）
MIN_STATIC_COLUMNS: List[str] = FEATURE_COLUMNS + SR_FEATURE_COLUMNS + ["base_pred"]


def _seq_features(
    g10: pd.DataFrame,
    preclose: float,
    vol_mean_row: pd.Series,
) -> np.ndarray:
    """计算 6 根 5 分钟线的序列特征，返回 (SEQ_LEN, len(SEQ_FEATURES)) 数组。

    g10       : 单日 10:00 前的 bar，按 time 升序，长度 = SEQ_LEN
    preclose  : 当日的前收盘价（T-1 收盘）
    vol_mean_row: 各位置（0..SEQ_LEN-1）的前 5 日同时段均量
    """
    g10 = g10.sort_values("time").reset_index(drop=True)
    open_day = g10["open"].iloc[0]
    closes = g10["close"].to_numpy(dtype=np.float64)
    volumes = g10["volume"].to_numpy(dtype=np.float64)
    highs = g10["high"].to_numpy(dtype=np.float64)
    lows = g10["low"].to_numpy(dtype=np.float64)

    prev = np.concatenate([[preclose], closes[:-1]])
    ret = closes / prev - 1.0
    vs_open = closes / open_day - 1.0
    vol_ratio = np.asarray(
        [volumes[i] / vol_mean_row.get(i, np.nan) for i in range(SEQ_LEN)],
        dtype=np.float64,
    )
    amplitude = (highs - lows) / open_day

    feats = np.stack([ret, vs_open, vol_ratio, amplitude], axis=1)
    return feats


def _daily_feature_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    """在日线数据上计算技术特征与压力位特征，并将特征日对齐到目标交易日。

    返回列：date（特征日 T-1）、target_date（交易日 T）、特征列。
    """
    daily = daily_df.sort_values("date").reset_index(drop=True)
    daily = compute_features(daily, window=20)
    daily = compute_sr_features(daily)
    daily["target_date"] = daily["date"].shift(-1)
    daily = daily.dropna(subset=["target_date"])

    # 剔除全 NaN 的静态列（如复权模式下的 turn）
    static_valid = [
        c for c in FEATURE_COLUMNS + SR_FEATURE_COLUMNS
        if c in daily.columns and not daily[c].isna().all()
    ]
    keep = ["date", "target_date"] + static_valid
    return daily[keep].dropna(subset=static_valid)


def _build_one_stock(
    code: str,
    frequency: str,
    base_preds: Optional[pd.DataFrame],
    macro_feats: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """构造单只股票的分钟样本。

    macro_feats : 宏观特征表（index=日期），无宏观数据时为 None；
                  基本面特征在内部自动检测加载。
    """
    from data import external_store
    from features.external import (
        compute_fundamental_features,
        merge_asof_previous_day,
    )

    min_df = store.load_min_bars(code, frequency)
    daily_df = store.load_bars(code)
    if min_df is None or min_df.empty or daily_df is None or daily_df.empty:
        logger.warning("跳过 %s：分钟线或日线数据缺失", code)
        return None

    # ---- 基本面事件特征（若有数据，T-1 日历日视角注入）----
    fundamental_feats: Optional[pd.DataFrame] = None
    events = external_store.load_fundamental_events(code)
    if events is not None and not events.empty:
        close_series = daily_df.set_index("date")["close"]
        fundamental_feats = compute_fundamental_features(
            events, daily_df["date"], close_series=close_series
        )

    # ---- 日线侧：T-1 特征，对齐到交易日 T ----
    daily = _daily_feature_frame(daily_df)
    if daily.empty:
        return None
    daily_lookup = daily.set_index("target_date")
    # T 日的前收盘价（T-1 收盘），用于第一根 bar 的相对涨跌
    daily_raw = daily_df.set_index("date")

    # ---- 分钟线侧：10:00 前的 bar，按位置统计前 5 日同时段均量 ----
    min_df = min_df.sort_values("time").reset_index(drop=True)
    min_df["hhmm"] = min_df["time"].dt.strftime("%H:%M")
    morning = min_df[min_df["hhmm"] <= CUTOFF_TIME].copy()
    if morning.empty:
        return None

    morning = morning.sort_values(["date", "time"])
    morning["pos"] = morning.groupby("date").cumcount()
    morning = morning[morning["pos"] < SEQ_LEN]

    vol_pivot = morning.pivot_table(
        index="date", columns="pos", values="volume", aggfunc="last"
    )
    # T 日行 = 前 5 个交易日的同位置均量（不含当日，避免泄漏）
    vol_mean = vol_pivot.rolling(5, min_periods=1).mean().shift(1)

    # ---- 逐日构造样本 ----
    rows: List[Dict] = []
    static_cols = [c for c in daily.columns if c not in ("date", "target_date", "preclose")]
    for d, g in min_df.groupby("date"):
        g10 = morning[morning["date"] == d]
        if len(g10) < SEQ_LEN:
            continue  # 上午数据不足（停牌/新股等）
        g10 = g10.sort_values("time").reset_index(drop=True)

        # 标签：10:00 收盘 → 15:00 收盘
        day_g = g.sort_values("time")
        close10 = g10["close"].iloc[-1]
        close_day = day_g["close"].iloc[-1]
        if close10 <= 0:
            continue
        label_rest = (close_day / close10 - 1.0) * 100.0

        # T-1 静态特征
        if d not in daily_lookup.index:
            continue
        stat_row = daily_lookup.loc[d]
        static_vals = stat_row[static_cols].to_dict()

        # T 日的前收盘价（注意：必须取 T 日的 preclose，而非 T-1 行的 preclose）
        if d not in daily_raw.index:
            continue
        preclose = float(daily_raw.loc[d, "preclose"])

        # 基座预测（级联特征）
        base_pred = None
        if base_preds is not None and len(base_preds) > 0:
            hit = base_preds.loc[
                (base_preds["date"] == d) & (base_preds["code"] == code), "base_pred"
            ]
            if len(hit) > 0:
                base_pred = float(hit.iloc[0])
        if base_pred is None:
            continue  # 无基座预测的日期不构造样本

        # 序列特征
        vol_mean_row = vol_mean.loc[d] if d in vol_mean.index else pd.Series(dtype=float)
        seq = _seq_features(g10, preclose, vol_mean_row)
        if np.isnan(seq).any():
            continue  # 历史均量不足等导致的缺失

        row = {
            "date": d,
            "code": code,
            "label_rest": label_rest,
            "label_cls_rest": int(label_rest > 0),
        }
        for i, f in enumerate(SEQ_FEATURES):
            for j in range(SEQ_LEN):
                row[f"seq_{j}_{f}"] = float(seq[j, i])
        for c in static_cols:
            row[c] = static_vals[c]
        row["base_pred"] = base_pred
        rows.append(row)

    if not rows:
        return None
    out = pd.DataFrame(rows)

    # ---- 外部特征注入（T-1 日历日视角：样本交易日 T 只能用 T-1 收盘前信息）----
    n_before = len(out.columns)
    if fundamental_feats is not None and not fundamental_feats.empty:
        out = merge_asof_previous_day(out, fundamental_feats)
    if macro_feats is not None and not macro_feats.empty:
        out = merge_asof_previous_day(out, macro_feats)
    new_cols = out.columns[n_before:]
    if len(new_cols) > 0:
        out[new_cols] = out[new_cols].fillna(0.0)

    return out


def build_min_samples(
    frequency: str = "5",
    base_preds: Optional[pd.DataFrame] = None,
    overwrite: bool = False,
) -> List[str]:
    """从 raw_min 分片流式构造分钟样本，写入 min_samples 分片。

    参数：
        frequency : 分钟线频率（5/15/30/60）
        base_preds: 基座模型预测表（date, code, base_pred），级联注入；
                    None 时跳过所有样本（级联依赖基座）
        overwrite : 是否重建已存在的样本分片

    返回：本次成功构建的股票代码列表。
    """
    codes = store.list_min_codes(frequency)
    if not codes:
        logger.warning("raw_min/%s 目录为空，请先运行 fetch-min", frequency)
        return []

    if base_preds is None or base_preds.empty:
        logger.warning("未提供基座预测 base_preds，无法构造样本（级联依赖基座）")
        return []
    base_preds = base_preds[["date", "code", "base_pred"]].dropna().reset_index(drop=True)

    # 宏观特征自动检测（盘前视角注入时用 T-1 日历日）
    from features.external import load_macro_features

    macro_feats = load_macro_features()
    if macro_feats is not None:
        logger.info("检测到宏观数据，将注入分钟样本（T-1 日历日视角）")

    built: List[str] = []
    skipped = 0
    for i, code in enumerate(codes):
        if store.min_sample_path(code, frequency).exists() and not overwrite:
            skipped += 1
            continue
        try:
            sample = _build_one_stock(code, frequency, base_preds, macro_feats)
        except Exception as exc:  # noqa: BLE001
            logger.error("构造 %s 分钟样本失败: %s", code, exc)
            continue
        if sample is not None and not sample.empty:
            store.save_min_samples(sample, code, frequency)
            built.append(code)
        if (i + 1) % 50 == 0:
            logger.info(
                "分钟样本进度 %d/%d（已构建 %d，跳过 %d）",
                i + 1, len(codes), len(built), skipped,
            )

    logger.info(
        "分钟样本构建完成：构建 %d 只，跳过 %d 只，共 %d 只",
        len(built), skipped, len(codes),
    )
    return built


# ---- 滚动窗口样本：当前 30 分钟窗口 → 预测下一个 30 分钟涨跌 ----

# 滚动步长 = 窗口长度（6 根 bar = 30 分钟，不重叠），每天约 7 条样本
ROLL_STEP = SEQ_LEN


def _window_seq_features(
    day_bars: pd.DataFrame,
    start_idx: int,
    preclose: float,
    vol_mean_by_time: pd.Series,
) -> np.ndarray:
    """计算任意连续 SEQ_LEN 根 bar 的序列特征（10:00 截面的泛化版）。

    day_bars        : 当日全部 bar，按 time 升序，index 0..n-1
    start_idx       : 窗口起始位置
    preclose        : 当日前收盘价
    vol_mean_by_time: Series，index=hhmm，前 5 日各时点均量
    """
    win = day_bars.iloc[start_idx:start_idx + SEQ_LEN]
    open_day = float(day_bars["open"].iloc[0])
    closes = win["close"].to_numpy(dtype=np.float64)
    highs = win["high"].to_numpy(dtype=np.float64)
    lows = win["low"].to_numpy(dtype=np.float64)
    volumes = win["volume"].to_numpy(dtype=np.float64)

    # ret：相对前一根 bar（窗口首根为当日开盘首根时相对前收）
    if start_idx == 0:
        prev = np.concatenate([[preclose], closes[:-1]])
    else:
        prev = np.concatenate(
            [[float(day_bars["close"].iloc[start_idx - 1])], closes[:-1]]
        )
    ret = closes / prev - 1.0

    vs_open = closes / open_day - 1.0

    # vol_ratio：相对前 5 日同 hhmm 时点均量
    hhmms = win["time"].dt.strftime("%H:%M")
    vol_ratio = np.asarray(
        [volumes[i] / vol_mean_by_time.get(t, np.nan) for i, t in enumerate(hhmms)],
        dtype=np.float64,
    )

    amplitude = (highs - lows) / open_day

    return np.stack([ret, vs_open, vol_ratio, amplitude], axis=1)


def build_roll_samples(
    code: str,
    frequency: str,
    base_preds: Optional[pd.DataFrame] = None,
    with_market: bool = True,
) -> Optional[pd.DataFrame]:
    """构造滚动窗口分钟样本：当前 30 分钟窗口预测下一个 30 分钟涨跌。

    - 滚动步长 6 根 bar（30 分钟，不重叠），每天约 7 条样本
    - 标签 label_rest：下一窗口末 bar 收盘 / 当前窗口末 bar 收盘 - 1（%）
    - 静态特征：T-1 日线 + 压力位 + base_pred，不注入宏观/基本面特征
    - with_market=True 时注入市场环境特征（指数环境+日历，T 日收盘可得），
      build_predict_input 推理时同样注入，保证训练/推理对齐
    - 午间休市：bar 序列天然连续，跨午休窗口（预测午后前 30 分钟）保留

    列结构与 build_min_samples 同构（MIN_SEQ_COLUMNS + 静态列 + label_rest 等），
    可直接复用 build_min_bundle / train_min_model。无数据时返回 None。
    """
    min_df = store.load_min_bars(code, frequency)
    daily_df = store.load_bars(code)
    if min_df is None or min_df.empty or daily_df is None or daily_df.empty:
        logger.warning("跳过 %s：分钟线或日线数据缺失", code)
        return None

    if base_preds is None or base_preds.empty:
        logger.warning("未提供基座预测 base_preds，无法构造滚动样本")
        return None
    base_preds = base_preds[["date", "code", "base_pred"]].dropna().reset_index(drop=True)

    daily = _daily_feature_frame(daily_df)
    if daily.empty:
        return None
    daily_lookup = daily.set_index("target_date")
    daily_raw = daily_df.set_index("date")

    # 全天各 hhmm 时点均量表（前 5 日滚动均值，不含当日，避免泄漏）
    m = min_df.sort_values("time").copy()
    m["hhmm"] = m["time"].dt.strftime("%H:%M")
    vol_pivot = m.pivot_table(index="date", columns="hhmm", values="volume", aggfunc="last")
    vol_mean = vol_pivot.rolling(5, min_periods=1).mean().shift(1)

    static_cols = [c for c in daily.columns if c not in ("date", "target_date", "preclose")]

    rows: List[Dict] = []
    for d, g in min_df.groupby("date"):
        g = g.sort_values("time").reset_index(drop=True)
        n = len(g)
        if n < SEQ_LEN * 2:
            continue  # 不足以构成一个窗口 + 预测目标
        if d not in daily_lookup.index or d not in daily_raw.index:
            continue
        stat_row = daily_lookup.loc[d]
        static_vals = stat_row[static_cols].to_dict()
        preclose = float(daily_raw.loc[d, "preclose"])
        vol_mean_row = vol_mean.loc[d] if d in vol_mean.index else pd.Series(dtype=float)

        # 基座预测（级联特征）
        hit = base_preds.loc[
            (base_preds["date"] == d) & (base_preds["code"] == code), "base_pred"
        ]
        if len(hit) == 0:
            continue  # 无基座预测的日期不构造样本（与日线样本一致）
        base_pred = float(hit.iloc[0])

        for start in range(0, n - SEQ_LEN * 2 + 1, ROLL_STEP):
            win_close = float(g["close"].iloc[start + SEQ_LEN - 1])
            tgt_close = float(g["close"].iloc[start + SEQ_LEN * 2 - 1])
            if win_close <= 0:
                continue
            label_rest = (tgt_close / win_close - 1.0) * 100.0

            seq = _window_seq_features(g, start, preclose, vol_mean_row)
            if np.isnan(seq).any():
                continue  # 历史均量不足等导致的缺失

            row = {
                "date": d,
                "code": code,
                "label_rest": label_rest,
                "label_cls_rest": int(label_rest > 0),
            }
            for i, f in enumerate(SEQ_FEATURES):
                for j in range(SEQ_LEN):
                    row[f"seq_{j}_{f}"] = float(seq[j, i])
            for c in static_cols:
                row[c] = static_vals[c]
            row["base_pred"] = base_pred
            rows.append(row)

    if not rows:
        return None
    out = pd.DataFrame(rows)
    # 注入市场环境特征（指数环境+日历效应，与 build_predict_input 对齐）
    if with_market:
        from features.market import MARKET_FEATURES, load_market_features

        market = load_market_features()
        if market is not None:
            out = out.merge(market.reset_index(), on="date", how="left")
            out[MARKET_FEATURES] = out[MARKET_FEATURES].fillna(0.0)
    return out


def build_predict_input(
    code: str,
    frequency: str,
    base_pred: Optional[float] = None,
    realtime_min: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    """构造单条预测输入（盘中推理用）：最新交易日 T 的 10:00 截面。

    从缓存读取分钟线与日线，取最新交易日 T 的 10:00 前 6 根 bar，
    结合 T-1 日线特征与基座预测，返回与训练样本同构的输入。

    参数：
        realtime_min : akshare 实时分钟线（项目 schema，可选）。提供时合并其
                       最新交易日的 bar 到缓存分钟线之上（不写缓存），
                       并在日线尚未覆盖当日时（盘中）自动构造当日临时日线行，
                       从而支持开盘 30 分钟即出信号。

    返回 dict：
        date/open_day/close10 : 交易日、当日开盘价、10:00 收盘价
        seq     : (1, SEQ_LEN, len(SEQ_FEATURES)) 序列特征
        static  : pd.Series，index 为静态特征列名
    数据不足（分钟线未到 10:00、日线缺失等）时返回 None。
    """
    min_df = store.load_min_bars(code, frequency)
    daily_df = store.load_bars(code)
    if min_df is None or min_df.empty or daily_df is None or daily_df.empty:
        logger.error("分钟线或日线数据缺失，请先运行 fetch-min / fetch")
        return None

    # 实时分钟线：合并其最新交易日的 bar（覆盖缓存当日旧数据）
    if realtime_min is not None and not realtime_min.empty:
        rt_date = realtime_min["date"].max()
        min_df = (
            pd.concat(
                [
                    min_df[min_df["date"] != rt_date],
                    realtime_min[realtime_min["date"] == rt_date],
                ],
                ignore_index=True,
            )
            .sort_values(["date", "time"])
            .reset_index(drop=True)
        )

    daily = _daily_feature_frame(daily_df)
    if daily.empty:
        logger.error("日线特征不足（需要至少 60+ 个交易日）")
        return None

    T = min_df["date"].max()
    daily_raw = daily_df.set_index("date")

    # 盘中场景：当日日线尚未生成，构造临时日线行（preclose=最后收盘日收盘价，
    # OHLC 用当日已成交 bar 填充）追加，使 target_date=T 的静态特征可用
    if T not in daily_raw.index:
        g_day = min_df[min_df["date"] == T]
        if g_day.empty:
            logger.error("交易日 %s 无分钟数据，无法构造当日日线", T.date())
            return None
        temp = pd.DataFrame(
            [
                {
                    "date": T,
                    "code": code,
                    "open": float(g_day["open"].iloc[0]),
                    "high": float(g_day["high"].max()),
                    "low": float(g_day["low"].min()),
                    "close": float(g_day["close"].iloc[-1]),
                    "preclose": float(daily_df["close"].iloc[-1]),  # 昨收 = 今前收
                    "volume": float(g_day["volume"].sum()),
                    "amount": float(g_day["amount"].sum()),
                    "turn": np.nan,
                    "pct_chg": np.nan,
                }
            ]
        )
        daily_df = (
            pd.concat([daily_df, temp], ignore_index=True)
            .sort_values("date")
            .reset_index(drop=True)
        )
        daily = _daily_feature_frame(daily_df)
        daily_raw = daily_df.set_index("date")

    if T not in daily_raw.index:
        logger.error("日线数据未覆盖交易日 %s", T.date())
        return None
    g = min_df[min_df["date"] == T]
    g10 = (
        g[g["time"].dt.strftime("%H:%M") <= CUTOFF_TIME]
        .sort_values("time")
        .reset_index(drop=True)
    )
    if len(g10) < SEQ_LEN:
        logger.error(
            "交易日 %s 的 10:00 截面不足 %d 根 bar（请 10:00 后运行）",
            T.date(), SEQ_LEN,
        )
        return None

    # 前 5 日同时段均量（不含当日）
    morning_all = min_df[
        min_df["time"].dt.strftime("%H:%M") <= CUTOFF_TIME
    ].copy()
    morning_all = morning_all.sort_values(["date", "time"])
    morning_all["pos"] = morning_all.groupby("date").cumcount()
    morning_all = morning_all[morning_all["pos"] < SEQ_LEN]
    vol_pivot = morning_all.pivot_table(
        index="date", columns="pos", values="volume", aggfunc="last"
    )
    vol_mean = vol_pivot.rolling(5, min_periods=1).mean().shift(1)
    vol_mean_row = vol_mean.loc[T] if T in vol_mean.index else pd.Series(dtype=float)

    # T-1 静态特征
    stat_row = daily[daily["target_date"] == T]
    if stat_row.empty:
        logger.error("日线特征未覆盖交易日 %s，请先运行 fetch --refresh", T.date())
        return None
    stat_row = stat_row.iloc[0]
    preclose = float(daily_raw.loc[T, "preclose"])  # T 日的前收盘价

    seq = _seq_features(g10, preclose, vol_mean_row)
    if np.isnan(seq).any():
        logger.error("序列特征存在缺失，无法预测")
        return None

    static_cols = [
        c for c in daily.columns
        if c not in ("date", "target_date", "preclose")
    ]
    static = stat_row[static_cols].astype(np.float32)
    static["base_pred"] = np.float32(base_pred if base_pred is not None else 0.0)

    # 市场环境特征（指数环境+日历，与 build_roll_samples 训练样本对齐）
    from features.market import MARKET_FEATURES, load_market_features

    market = load_market_features()
    if market is not None and T in market.index:
        for c in MARKET_FEATURES:
            static[c] = float(market.loc[T, c])

    return {
        "date": T,
        "open_day": float(g10["open"].iloc[0]),
        "close10": float(g10["close"].iloc[-1]),
        "seq": seq.astype(np.float32).reshape(1, SEQ_LEN, len(SEQ_FEATURES)),
        "static": static,
    }
