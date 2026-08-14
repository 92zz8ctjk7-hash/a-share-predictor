"""回测编排：数据准备 → 按窗口训练（含 checkpoint）→ 样本外网格回测 → 多窗口汇总。

流程（对每只股票、每个窗口）：
1. 取窗口内样本（窗口起点 = 样本最晚日期往前 N 年）
2. 窗口内按时间切分：前 train_ratio 训练（内部再切训练/验证），
   后 (1-train_ratio) 为样本外回测段——5y 窗口即「约 4 年训练 + 1 年回测」
3. 训练模型（best checkpoint 自动保存），对回测段逐日生成预测信号
4. 网格策略回测：T 日收盘出信号、T+1 开盘撮合、T+1 制度、整手交易
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import DATA_DIR, Config, cfg
from features.builder import build_dataset

logger = logging.getLogger(__name__)

# 窗口标签 → 年数
_WINDOW_YEARS = {"5y": 5, "3y": 3, "2y": 2, "1y": 1}


def ensure_data(code: str, min_years: float = 4.5) -> pd.DataFrame:
    """读取 raw 分片；不存在或历史长度不足时全量拉取覆盖。"""
    from data import store
    from data.fetcher import fetch_stock

    bars = store.load_bars(code)
    need_start = (date.today() - pd.Timedelta(days=int(min_years * 365))).isoformat()
    if bars is None or bars.empty or str(bars["date"].min().date()) > need_start:
        logger.info("本地数据不足，全量拉取 %s（%s 起）", code, need_start)
        # fetch_stock 的 use_cache=False 会同时跳过保存，
        # 因此先删旧分片再走正常缓存路径（拉取后自动覆盖保存）
        cache_file = store.raw_path(code)
        if cache_file.exists():
            cache_file.unlink()
        bars = fetch_stock(
            code=code,
            start_date=cfg.start_date,
            end_date=date.today().isoformat(),
            frequency="d",
            adjust=cfg.adjust,
            use_cache=True,
        )
    return bars


def _window_start(max_date: pd.Timestamp, window: str) -> pd.Timestamp:
    years = _WINDOW_YEARS[window]
    return max_date - pd.DateOffset(years=years)


def _train_and_predict(
    sample_win: pd.DataFrame,
    model_name: str,
    ckpt_dir: Path,
    epochs: Optional[int],
    save_every: int,
    train_ratio: float,
) -> pd.DataFrame:
    """窗口内训练模型并对回测段推理，返回信号表（date/pred_reg/prob_up）。

    训练段内部按 85/15 再切训练/验证（验证用于 best checkpoint 选择）。
    """
    from features.dataset import DataBundle, StockDataset, _feature_cols_of
    from models.train import (
        load_nn_checkpoint,
        nn_predict,
        train_model,
    )
    from config import Config

    feature_names = _feature_cols_of(sample_win)
    sub = sample_win.sort_values("date").reset_index(drop=True)
    n = len(sub)
    train_end = int(n * train_ratio)
    train_part, test_part = sub.iloc[:train_end], sub.iloc[train_end:]
    if train_end < 30 or len(test_part) < 10:
        return pd.DataFrame(columns=["date", "pred_reg", "prob_up"])

    # 训练段内部切验证（时间顺序）
    inner_end = int(len(train_part) * 0.85)
    tr, va = train_part.iloc[:inner_end], train_part.iloc[inner_end:]

    mode = "seq" if model_name == "lstm" else "flat"
    seq_len = cfg.seq_len

    def _ds(df: pd.DataFrame) -> StockDataset:
        return StockDataset(
            df[feature_names].to_numpy(dtype=np.float32),
            df["label_reg"].to_numpy(dtype=np.float32),
            df["label_cls"].to_numpy(dtype=np.int64),
            seq_len=seq_len,
            mode=mode,
        )

    bundle = DataBundle(
        train=_ds(tr),
        valid=_ds(va),
        test=_ds(test_part),  # 占位，不参与训练
        feature_names=feature_names,
        train_df=tr.reset_index(drop=True),
        valid_df=va.reset_index(drop=True),
        test_df=test_part.reset_index(drop=True),
    )

    eff_cfg = Config(
        start_date=cfg.start_date, end_date=cfg.end_date,
        horizon=cfg.horizon, window=cfg.window,
        model=model_name,
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        seq_len=seq_len, dropout=cfg.dropout,
        epochs=epochs if epochs is not None else cfg.epochs,
        batch_size=cfg.batch_size, lr=cfg.lr, seed=cfg.seed,
        device=cfg.device,
    )
    train_model(bundle, eff_cfg, ckpt_dir=str(ckpt_dir), save_every=save_every)

    # ---- 对回测段推理（seq 模式需带入切分点前的 seq_len-1 个样本构造序列）----
    if model_name == "baseline":
        import joblib

        model = joblib.load(ckpt_dir / "best.joblib")
        X = test_part[feature_names].to_numpy(dtype=np.float32)
        pred_reg, _, prob_up = model.predict(X)
        dates = test_part["date"].to_numpy()
    else:
        net, meta = load_nn_checkpoint(str(ckpt_dir / "best.pt"))
        # 特征列以 checkpoint 保存的顺序为准
        cols = meta["feature_names"]
        if mode == "seq":
            # 上下文仅取特征列（避免 date/code 等非数值列混入）
            context = train_part[feature_names].tail(seq_len - 1)
            seq_src = pd.concat([context, test_part[feature_names]], ignore_index=True)
            X = seq_src.to_numpy(dtype=np.float32)
            pred_reg, prob_up = nn_predict(net, X, mode="seq", seq_len=seq_len)
            dates = test_part["date"].to_numpy()
        else:
            X = test_part[cols].to_numpy(dtype=np.float32)
            pred_reg, prob_up = nn_predict(net, X, mode="flat")
            dates = test_part["date"].to_numpy()

    return pd.DataFrame(
        {"date": pd.to_datetime(dates), "pred_reg": pred_reg, "prob_up": prob_up}
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _train_intraday_and_predict(
    code: str,
    frequency: str,
    train_ratio: float,
    ckpt_dir: Path,
    epochs: Optional[int],
):
    """训练 IntradayLSTM 分钟模型并对回测段推理，返回 (signals, min_df)。

    分钟样本按时间切分：前 (train_ratio - 0.1) 训练 + 0.1 验证 + 后 (1-train_ratio)
    回测（防前视）。回测段信号 timing="close"（10:00 信息已知，当日收盘可撮合）。
    prob_up = sigmoid 分类输出，语义为「当日收盘 > 10:00 价」的概率。
    无分钟样本时返回 (None, 空 DataFrame)。
    """
    from data import store
    from features.min_dataset import build_min_bundle
    from models.train import save_min_model, train_min_model

    all_min = store.load_all_min_samples(frequency)
    if all_min is None or all_min.empty:
        return None, pd.DataFrame()
    min_df = all_min[all_min["code"] == code].sort_values("date").reset_index(drop=True)
    if len(min_df) < 60:
        logger.warning("%s 分钟样本不足（%d），无法训练分钟模型", code, len(min_df))
        return None, min_df

    # 内部验证占 0.1，回测段 = 1 - train_ratio
    valid_ratio = 0.10
    bundle = build_min_bundle(
        min_df,
        train_ratio=max(train_ratio - valid_ratio, 0.1),
        valid_ratio=valid_ratio,
    )

    eff_cfg = Config(
        model="intraday_lstm",
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        epochs=epochs if epochs is not None else cfg.epochs,
        batch_size=cfg.batch_size, lr=cfg.lr, seed=cfg.seed, device=cfg.device,
        start_date=cfg.start_date, end_date=cfg.end_date,
        horizon=cfg.horizon, window=cfg.window,
        seq_len=cfg.seq_len,
    )
    logger.info(
        "=== %s | 分钟模型 IntradayLSTM | 样本 %d（训练 %d/验证 %d/回测 %d）===",
        code, len(min_df), len(bundle.train), len(bundle.valid), len(bundle.test),
    )
    trainer, metrics = train_min_model(bundle, eff_cfg)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_min_model(trainer, bundle, str(ckpt_dir / "intraday_best.pt"))

    reg_pred, _, cls_score = trainer.predict(bundle, "test")
    dates = bundle.test_df["date"].to_numpy()
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "pred_reg": reg_pred,
            "prob_up": cls_score,
            "timing": "close",
        }
    )
    return signals, min_df


def _merge_with_base(
    signals_min: pd.DataFrame, bars_bt: pd.DataFrame, code: str
) -> pd.DataFrame:
    """级联回退：回测段内无分钟信号的交易日，用基座 base_pred 补 signal。

    分钟信号 timing="close"；回退基座信号 timing="next_open"（日线视角）。
    prob_up = sigmoid(base_pred)。
    """
    base_path = DATA_DIR / "meta" / "base_preds.parquet"
    min_dates = set(pd.to_datetime(signals_min["date"]))
    missing = [d for d in bars_bt["date"] if d not in min_dates]
    if not missing or not base_path.exists():
        return signals_min

    base = pd.read_parquet(base_path)
    fb = base[(base["code"] == code) & (pd.to_datetime(base["date"]).isin(missing))]
    if fb.empty:
        return signals_min

    fb = pd.DataFrame(
        {
            "date": pd.to_datetime(fb["date"]),
            "pred_reg": fb["base_pred"].to_numpy(dtype=np.float64),
            "prob_up": _sigmoid(fb["base_pred"]),
            "timing": "next_open",
        }
    )
    out = pd.concat([signals_min, fb], ignore_index=True)
    return out.sort_values("date").reset_index(drop=True)


def _predict_with_pretrained(bars: pd.DataFrame, model_name: str = "lstm") -> pd.DataFrame:
    """加载已训练的基座 checkpoint 全量推理，返回 (date, pred_reg, prob_up, timing)。

    特征对齐训练时口径：FEATURE_COLUMNS（剔除全 NaN 列）+ 宏观特征。
    timing 固定 next_open：T 日收盘信号 → T+1 开盘撮合。
    """
    import torch

    from features.builder import FEATURE_COLUMNS, compute_features
    from features.external import load_macro_features
    from models.nn import LSTM
    from models.train import nn_predict

    model_path = DATA_DIR / "models" / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {model_path}，请先训练")

    df = compute_features(
        bars.sort_values("date").reset_index(drop=True), window=cfg.window
    )
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns and not df[c].isna().all()]
    macro = load_macro_features()
    if macro is not None:
        df = df.merge(macro.reset_index(), on="date", how="left")
        macro_cols = list(macro.columns)
        df[macro_cols] = df[macro_cols].fillna(0.0)
        feat_cols = feat_cols + macro_cols

    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    w = [v for k, v in state.items() if "weight_ih_l0" in k][0]
    if w.shape[1] != len(feat_cols):
        raise ValueError(
            f"特征维度 {len(feat_cols)} 与 checkpoint 输入维度 {w.shape[1]} 不匹配"
        )

    net = LSTM(
        input_dim=len(feat_cols),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    net.load_state_dict(state)

    feats = df[feat_cols].to_numpy(dtype=np.float32)
    pred_reg, prob_up = nn_predict(net, feats, mode="seq", seq_len=cfg.seq_len)
    valid_dates = df["date"].iloc[cfg.seq_len - 1:].reset_index(drop=True)
    logger.info("pretrained 推理完成：%d 条信号（%s）", len(pred_reg), model_path.name)
    return pd.DataFrame(
        {
            "date": valid_dates.iloc[: len(pred_reg)],
            "pred_reg": pred_reg,
            "prob_up": prob_up,
            "timing": "next_open",
        }
    )


def _try_plot(code: str, window: str) -> None:
    """回测后自动绘制资金曲线（尽力而为，失败不阻塞回测）。"""
    try:
        from backtest.plot import plot_equity

        plot_equity(code, window)
    except Exception as exc:  # noqa: BLE001
        logger.warning("资金曲线自动绘图失败：%s", exc)


def run_backtest(
    codes: List[str],
    capital: float = 100000.0,
    windows: Optional[List[str]] = None,
    model: str = "baseline",
    grid_n: int = 10,
    range_pct: float = 0.20,
    gate_on: bool = True,
    gate_threshold: float = 0.5,
    commission: float = 2.5e-4,
    min_commission: float = 5.0,
    stamp_tax: float = 5e-4,
    slippage: float = 1e-3,
    train_ratio: float = 0.8,
    epochs: Optional[int] = None,
    save_every: int = 10,
    keep_curves: bool = False,
    signal_source: str = "base",
    min_frequency: str = "5",
    pretrained: bool = False,
) -> pd.DataFrame:
    """多窗口网格回测主入口，返回汇总报告 DataFrame。

    参数：
        signal_source : "base"（日线模型信号，现有行为）或 "cascade"
                        （分钟模型信号 timing=close，缺失日回退基座）
        min_frequency : cascade 模式下的分钟线频率
        pretrained    : True 时直接加载已有 checkpoint（cache/models/{model}.pt）
                        对窗口全段推理回测，不重新训练

    每行 = (code, window) 的回测绩效；报告同时保存到
    data/meta/backtest_report.csv，资金曲线保存到 data/meta/equity_{code}_{window}.csv。
    """
    from backtest.engine import CostConfig, run_engine
    from backtest.strategy import GridStrategy

    windows = windows or cfg.bt_windows
    cost = CostConfig(
        commission_rate=commission,
        min_commission=min_commission,
        stamp_tax=stamp_tax,
        slippage=slippage,
    )
    meta_dir = DATA_DIR / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for code in codes:
        bars = ensure_data(code)
        if bars is None or bars.empty:
            logger.error("无法获取 %s 的行情数据，跳过", code)
            continue

        # ---- pretrained 模式：加载已有 checkpoint 直接推理回测（不重训）----
        if pretrained:
            try:
                signals_all = _predict_with_pretrained(bars, model)
            except Exception as exc:  # noqa: BLE001
                logger.error("%s pretrained 推理失败：%s", code, exc)
                continue

            max_bar_date = bars["date"].max()
            for window in windows:
                if window not in _WINDOW_YEARS:
                    logger.warning("未知窗口 %s，跳过", window)
                    continue
                win_start = _window_start(max_bar_date, window)
                bars_bt = bars[bars["date"] >= win_start].reset_index(drop=True)
                signals_bt = signals_all[signals_all["date"] >= win_start]
                logger.info(
                    "=== %s | 窗口 %s | pretrained 回测：%s 起 %d 交易日 ===",
                    code, window, win_start.date(), len(bars_bt),
                )
                strategy = GridStrategy(
                    grid_n=grid_n, range_pct=range_pct,
                    gate_on=gate_on, gate_threshold=gate_threshold,
                )
                result = run_engine(
                    bars=bars_bt, signals=signals_bt, strategy=strategy,
                    capital=capital, cost=cost,
                )
                if "error" in result.stats:
                    logger.warning("%s %s 回测失败：%s", code, window, result.stats["error"])
                    continue
                rows.append(
                    {"code": code, "window": window, "capital": capital, **result.stats}
                )
                if keep_curves:
                    if result.equity_curve is not None:
                        result.equity_curve.to_csv(
                            meta_dir / f"equity_{code}_{window}.csv", index=False
                        )
                    if result.trades is not None and not result.trades.empty:
                        result.trades.to_csv(
                            meta_dir / f"trades_{code}_{window}.csv", index=False
                        )
                    _try_plot(code, window)
            continue

        sample = build_dataset(bars, window=cfg.window, horizon=cfg.horizon)
        if sample.empty and signal_source != "cascade":
            logger.error("%s 样本构建为空，跳过", code)
            continue

        # ---- 级联模式：分钟模型信号（回测段由分钟样本决定，忽略 windows）----
        if signal_source == "cascade":
            ckpt_dir = DATA_DIR / "models" / "checkpoints" / f"{code}_intraday"
            signals_min, min_df = _train_intraday_and_predict(
                code, min_frequency, train_ratio, ckpt_dir, epochs
            )
            if signals_min is None or signals_min.empty:
                logger.warning("%s 无可用分钟信号，跳过 cascade", code)
                continue

            bt_start = signals_min["date"].min()
            bars_bt = bars[bars["date"] >= bt_start].reset_index(drop=True)
            signals = _merge_with_base(signals_min, bars_bt, code)

            strategy = GridStrategy(
                grid_n=grid_n, range_pct=range_pct,
                gate_on=gate_on, gate_threshold=gate_threshold,
            )
            result = run_engine(
                bars=bars_bt, signals=signals, strategy=strategy,
                capital=capital, cost=cost,
            )
            if "error" in result.stats:
                logger.warning("%s cascade 回测失败：%s", code, result.stats["error"])
                continue

            rows.append(
                {
                    "code": code, "window": f"min_{min_frequency}",
                    "capital": capital, **result.stats,
                }
            )
            if keep_curves and result.equity_curve is not None:
                curve_path = meta_dir / f"equity_{code}_min_{min_frequency}.csv"
                result.equity_curve.to_csv(curve_path, index=False)
                _try_plot(code, f"min_{min_frequency}")
            continue

        max_date = sample["date"].max()
        for window in windows:
            if window not in _WINDOW_YEARS:
                logger.warning("未知窗口 %s，跳过", window)
                continue

            sub = sample[sample["date"] >= _window_start(max_date, window)]
            if len(sub) < 120:
                logger.warning("%s %s 窗口样本不足（%d），跳过", code, window, len(sub))
                continue

            ckpt_dir = DATA_DIR / "models" / "checkpoints" / f"{code}_{model}_{window}"
            logger.info(
                "=== %s | 窗口 %s | 样本 %d | 模型 %s ===",
                code, window, len(sub), model,
            )
            signals = _train_and_predict(
                sub, model, ckpt_dir, epochs, save_every, train_ratio
            )

            # 回测段行情：训练/回测切分点起（含切分点当日，保证首日可出信号）
            n = len(sub.sort_values("date").reset_index(drop=True))
            split_date = sub.sort_values("date").iloc[int(n * train_ratio)]["date"]
            bars_bt = bars[bars["date"] >= split_date].reset_index(drop=True)

            strategy = GridStrategy(
                grid_n=grid_n,
                range_pct=range_pct,
                gate_on=gate_on,
                gate_threshold=gate_threshold,
            )
            result = run_engine(
                bars=bars_bt,
                signals=signals,
                strategy=strategy,
                capital=capital,
                cost=cost,
            )

            if "error" in result.stats:
                logger.warning("%s %s 回测失败：%s", code, window, result.stats["error"])
                continue

            row = {"code": code, "window": window, "capital": capital, **result.stats}
            rows.append(row)

            if keep_curves and result.equity_curve is not None:
                curve_path = meta_dir / f"equity_{code}_{window}.csv"
                result.equity_curve.to_csv(curve_path, index=False)
                logger.info("资金曲线已保存到 %s", curve_path)
                _try_plot(code, window)

    report = pd.DataFrame(rows)
    if report.empty:
        logger.warning("无任何回测结果")
        return report

    out = meta_dir / "backtest_report.csv"
    report.to_csv(out, index=False)
    logger.info("回测报告已保存到 %s（%d 行）", out, len(report))
    return report
