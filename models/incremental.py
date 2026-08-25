"""增量更新：刷新最新数据 → 重建样本 → 滚动窗口重训基座与分钟模型。

滚动窗口重训 = 训练时按日期过滤样本到最近窗口（如基座近 2 年、分钟近 12 个月）
重新训练，使模型贴合近期行情；可选 --resume 加载已有 checkpoint warm-start。

典型用法（收盘后定时运行，随后 min-predict 即用新模型实时预测）：
    python main.py incremental-update --codes sz.000100 --base-window 2y --min-window 12m
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import Config, DATA_DIR, cfg
from data import store

logger = logging.getLogger(__name__)


def _parse_window(spec: str) -> pd.DateOffset:
    """解析窗口描述：2y / 12m / 90d -> pd.DateOffset。"""
    m = re.fullmatch(r"(\d+)\s*([ymd])", str(spec).strip().lower())
    if not m:
        raise ValueError(f"无法解析窗口 '{spec}'，应为 数字+单位，如 2y / 12m / 90d")
    n, unit = int(m.group(1)), m.group(2)
    if unit == "y":
        return pd.DateOffset(years=n)
    if unit == "m":
        return pd.DateOffset(months=n)
    return pd.DateOffset(days=n)


def _load_base_codes() -> List[str]:
    """基座股票池：优先沪深300清单，否则用已缓存 raw 的全部代码。"""
    p = DATA_DIR / "meta" / "hs300_codes.txt"
    if p.exists():
        codes = [line.strip() for line in p.read_text().splitlines() if line.strip()]
        if codes:
            return codes
    return store.list_raw_codes()


def _eff_cfg(model: str, epochs: int) -> Config:
    """按 CLI 参数构造与训练一致的有效配置。"""
    return Config(
        start_date=cfg.start_date, end_date=cfg.end_date,
        horizon=cfg.horizon, window=cfg.window,
        train_ratio=cfg.train_ratio, valid_ratio=cfg.valid_ratio,
        model=model,
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, seq_len=cfg.seq_len,
        dropout=cfg.dropout, epochs=epochs, batch_size=cfg.batch_size,
        lr=cfg.lr, seed=cfg.seed, device=cfg.device,
        default_codes=cfg.default_codes,
    )


def _maybe_load_state(resume: bool, path) -> Optional[dict]:
    """resume 时加载已有 checkpoint 的 state_dict（兼容裸 state_dict 与带 meta 的 ckpt）。"""
    if not resume or not path.exists():
        return None
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt
    except Exception as exc:  # noqa: BLE001
        logger.warning("加载 %s 失败: %s", path, exc)
        return None


def _train_base_window(sample, window_spec, model_name, epochs, resume) -> Tuple[int, object]:
    """窗口过滤样本训练基座模型并保存，返回 (窗口样本数, bundle)。"""
    from features.dataset import build_bundle
    from models.train import save_model, train_model

    offset = _parse_window(window_spec)
    max_date = sample["date"].max()
    cutoff = max_date - offset
    sub = sample[sample["date"] >= cutoff].reset_index(drop=True)
    logger.info("基座窗口训练：近 %s（%s 起）%d 条样本", window_spec, cutoff.date(), len(sub))

    mode = "seq" if model_name == "lstm" else "flat"
    bundle = build_bundle(
        sub, train_ratio=cfg.train_ratio, valid_ratio=cfg.valid_ratio,
        seq_len=cfg.seq_len, mode=mode,
        persist_scaler=True,  # 持久化标准化参数，供独立推理路径对齐
    )
    eff_cfg = _eff_cfg(model_name, epochs)
    init_state = _maybe_load_state(resume, DATA_DIR / "models" / f"{model_name}.pt")
    try:
        model_info = train_model(bundle, eff_cfg, init_state_dict=init_state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("warm-start 失败（%s），从头训练", exc)
        model_info = train_model(bundle, eff_cfg)
    save_model(model_info, str(DATA_DIR / "models" / f"{model_name}.pt"))
    return len(sub), bundle


def _train_nextday_window(sample, window_spec, epochs, resume) -> Tuple[int, object]:
    """训练次日预测模型（horizon=1），保存为 lstm_nextday.pt。

    样本标签从 close 列推导次日收益（不复用 horizon=5 的 label_reg）：
    label_1d = close[t+1]/close[t] - 1（按股票分组）。特征/窗口/标准化与基座一致，
    故标准化参数与基座相同（同特征同日期切分），推理复用同一 scaler。
    """
    from features.dataset import build_bundle
    from models.train import save_model, train_model

    offset = _parse_window(window_spec)
    max_date = sample["date"].max()
    cutoff = max_date - offset
    sub = sample[sample["date"] >= cutoff].reset_index(drop=True).copy()

    # 次日标签：按股票分组，次日收盘/当日收盘 - 1（%）
    sub = sub.sort_values(["code", "date"]).reset_index(drop=True)
    nxt_close = sub.groupby("code")["close"].shift(-1)
    sub["label_reg"] = (nxt_close / sub["close"] - 1) * 100.0
    sub["label_cls"] = (sub["label_reg"] > 0).astype(int)
    sub = sub.dropna(subset=["label_reg"]).reset_index(drop=True)
    logger.info("次日模型窗口训练：近 %s（%s 起）%d 条样本", window_spec, cutoff.date(), len(sub))

    bundle = build_bundle(
        sub, train_ratio=cfg.train_ratio, valid_ratio=cfg.valid_ratio,
        seq_len=cfg.seq_len, mode="seq",
        persist_scaler=False,  # 复用基座同一份标准化（同特征同切分）
    )
    eff_cfg = _eff_cfg("lstm", epochs)
    init_state = _maybe_load_state(resume, DATA_DIR / "models" / "lstm_nextday.pt")
    try:
        model_info = train_model(bundle, eff_cfg, init_state_dict=init_state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("warm-start 失败（%s），从头训练", exc)
        model_info = train_model(bundle, eff_cfg)
    save_model(model_info, str(DATA_DIR / "models" / "lstm_nextday.pt"))
    return len(sub), bundle


def _train_min_window(min_sample, window_spec, frequency, epochs, resume) -> int:
    """窗口过滤滚动分钟样本训练 IntradayLSTM 并保存，返回窗口样本数。

    样本标签语义为「下一个 30 分钟涨跌幅」（滚动窗口构造，见 build_roll_samples）。
    """
    from features.min_dataset import build_min_bundle
    from models.train import save_min_model, train_min_model

    offset = _parse_window(window_spec)
    max_date = min_sample["date"].max()
    cutoff = max_date - offset
    sub = min_sample[min_sample["date"] >= cutoff].reset_index(drop=True)
    logger.info("分钟窗口训练：近 %s（%s 起）%d 条样本", window_spec, cutoff.date(), len(sub))

    bundle = build_min_bundle(sub, train_ratio=cfg.train_ratio, valid_ratio=cfg.valid_ratio)
    eff_cfg = _eff_cfg("intraday_lstm", epochs)
    init_state = _maybe_load_state(
        resume, DATA_DIR / "models" / f"intraday_lstm_{frequency}.pt"
    )
    try:
        trainer, _ = train_min_model(bundle, eff_cfg, init_state_dict=init_state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("warm-start 失败（%s），从头训练", exc)
        trainer, _ = train_min_model(bundle, eff_cfg)
    save_min_model(trainer, bundle, str(DATA_DIR / "models" / f"intraday_lstm_{frequency}.pt"))
    return len(sub)


def _aux_base_preds(codes: List[str], model_name: str = "lstm") -> Optional[pd.DataFrame]:
    """用已保存的基座模型对辅助训练股推理，返回 (date, code, base_pred)。

    辅助股不在基座样本池（沪深300），predict_base_all 覆盖不到，
    这里逐股用 checkpoint 全量推理（特征口径与训练一致）。
    """
    if not codes:
        return None
    from backtest.run import _predict_with_pretrained
    from data import store

    frames = []
    for c in codes:
        bars = store.load_bars(c)
        if bars is None or bars.empty:
            logger.warning("辅助股 %s 无日线数据，跳过基座推理", c)
            continue
        try:
            sig = _predict_with_pretrained(bars, model_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("辅助股 %s 基座推理失败: %s", c, exc)
            continue
        frames.append(pd.DataFrame(
            {"date": sig["date"], "code": c, "base_pred": sig["pred_reg"]}
        ))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def run_incremental_update(
    codes: List[str],
    base_window: str = "2y",
    min_window: str = "12m",
    base_model: str = "lstm",
    base_epochs: int = 10,
    min_epochs: int = 30,
    resume: bool = False,
    frequency: str = "5",
) -> Dict:
    """一键增量更新全流程，返回各阶段摘要。

    辅助训练股（cfg.min_aux_codes，面板板块）：刷新日线+分钟线、生成基座预测、
    构建滚动样本参与分钟模型联合训练，但不参与 serving。
    """
    from data import store
    from data.fetcher import BaoStockClient, fetch_minutes, fetch_stock
    from features.builder import build_samples_to_store
    from features.min_builder import build_roll_samples
    from models.train import predict_base_all

    today = date.today().isoformat()
    base_codes = _load_base_codes()
    aux_codes = [c for c in cfg.min_aux_codes if c not in codes]
    # 板块反转参考股（仅刷日线，不参与训练）
    from serving.intraday_signal import PEER_NAMES

    peer_codes = [
        c for c in PEER_NAMES
        if c not in base_codes and c not in aux_codes and c not in codes
    ]
    summary: Dict = {}

    # 1. 数据刷新（串行 baostock，避免会话互踢）
    logger.info(
        "[1/6] 刷新数据：基座 %d 只日线 + %d 只分钟线（含辅助股 %d 只）+ 板块参考股 %d 只日线",
        len(base_codes), len(codes) + len(aux_codes), len(aux_codes), len(peer_codes),
    )
    with BaoStockClient() as client:
        for c in base_codes + aux_codes + peer_codes:
            try:
                fetch_stock(c, cfg.start_date, today, frequency="d", adjust=cfg.adjust,
                            use_cache=True, refresh=True, client=client)
            except Exception as exc:  # noqa: BLE001
                logger.error("刷新日线 %s 失败: %s", c, exc)
        for c in codes + aux_codes:
            try:
                fetch_minutes(c, cfg.start_date, today, frequency=frequency,
                              refresh=True, client=client)
            except Exception as exc:  # noqa: BLE001
                logger.error("刷新分钟线 %s 失败: %s", c, exc)

    # 2. 日线样本重建
    logger.info("[2/6] 重建日线样本（基座 %d 只）", len(base_codes))
    build_samples_to_store(window=cfg.window, horizon=cfg.horizon,
                           overwrite=True, codes=base_codes)

    # 3. 基座窗口训练
    logger.info("[3/6] 基座窗口训练")
    sample = store.load_all_samples()
    n_base, bundle = _train_base_window(sample, base_window, base_model, base_epochs, resume)
    summary["base_window_samples"] = n_base

    # 3.5 次日预测模型（horizon=1，供隔夜推送）
    try:
        n_nextday, _ = _train_nextday_window(sample, base_window, base_epochs, resume)
        summary["nextday_samples"] = n_nextday
    except Exception as exc:  # noqa: BLE001
        logger.warning("次日模型训练失败（不影响主流程）: %s", exc)

    # 4. base-predict
    logger.info("[4/6] 基座推理生成 base_preds")
    model_path = DATA_DIR / "models" / f"{base_model}.pt"
    eff_cfg = _eff_cfg(base_model, base_epochs)
    base_preds = predict_base_all(bundle, eff_cfg, str(model_path))
    # 辅助训练股不在基座样本池，用已保存模型单独推理后合并
    aux_preds = _aux_base_preds(aux_codes, base_model)
    if aux_preds is not None and not aux_preds.empty:
        base_preds = pd.concat(
            [base_preds[~base_preds["code"].isin(aux_codes)], aux_preds],
            ignore_index=True,
        )
        logger.info("辅助股基座预测已合并：%d 条", len(aux_preds))
    bp_path = DATA_DIR / "meta" / "base_preds.parquet"
    bp_path.parent.mkdir(parents=True, exist_ok=True)
    base_preds.to_parquet(bp_path, index=False)

    # 5. 滚动分钟样本重建（当前 30 分钟 → 预测下一个 30 分钟；含辅助股）
    logger.info("[5/6] 重建滚动分钟样本")
    for code in codes + aux_codes:
        roll = build_roll_samples(code, frequency, base_preds)
        if roll is not None and not roll.empty:
            store.save_roll_samples(roll, code, frequency)
            logger.info("%s 滚动样本构建完成：%d 条", code, len(roll))
        else:
            logger.warning("%s 滚动样本构建为空", code)

    # 6. 分钟窗口训练（滚动样本）
    logger.info("[6/6] 分钟窗口训练（滚动样本）")
    min_sample = store.load_all_roll_samples(frequency)
    n_min = _train_min_window(min_sample, min_window, frequency, min_epochs, resume)
    summary["min_window_samples"] = n_min

    # 7. logistic 门控重训（生产策略的买入门控，秒级）
    try:
        from rl_gate.gate import train_gate
        from rl_gate.opportunity import build_opportunities, get_features

        logger.info("[+] 重训 logistic 门控")
        base_codes = _load_base_codes()
        feats_list = get_features(True)
        opp = build_opportunities(base_codes, with_market=True)
        gate_bundle = train_gate(opp, feats_list)
        summary["gate_auc"] = round(gate_bundle["auc"], 4)
    except Exception as exc:  # noqa: BLE001
        logger.warning("logistic 门控重训失败（不影响主流程）: %s", exc)

    logger.info("增量更新完成：%s", summary)
    return summary
