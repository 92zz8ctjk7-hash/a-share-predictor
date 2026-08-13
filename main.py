"""CLI 入口：A 股涨跌幅预测系统。

用法：
    python main.py fetch    --all                          # 全量拉取全市场（分片存储）
    python main.py fetch    --all --refresh                # 增量更新到最新交易日
    python main.py fetch    --codes sh.600000,sz.000001    # 小批量拉取
    python main.py fetch-min --codes sh.600000 --frequency 5   # 部分股票分钟线
    python main.py features --window 20 --horizon 5        # 流式构建全量样本
    python main.py train    --model lstm --epochs 30
    python main.py predict  --code sh.600000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from config import Config, DATA_DIR, ROOT, cfg

logger = logging.getLogger("ashare")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_codes(codes_arg: str):
    codes = [c.strip() for c in codes_arg.split(",") if c.strip()]
    return codes


# ---- 子命令实现 ----


def cmd_fetch(args) -> None:
    from data.fetcher import fetch_all, fetch_many

    start = args.start or cfg.start_date
    end = args.end or cfg.end_date or date.today().isoformat()
    adjust = args.adjust or cfg.adjust
    frequency = args.frequency or cfg.frequency

    if args.all:
        logger.info(
            "开始全量拉取：区间 %s ~ %s，frequency=%s adjust=%s refresh=%s",
            start, end, frequency, adjust, args.refresh,
        )
        stats = fetch_all(
            start_date=start,
            end_date=end,
            frequency=frequency,
            adjust=adjust,
            sleep=args.sleep if args.sleep is not None else cfg.fetch_sleep,
            include_delisted=not args.no_delisted,
            refresh=args.refresh,
        )
        logger.info("拉取统计：%s", stats)
        return

    codes = parse_codes(args.codes) if args.codes else cfg.default_codes
    logger.info("开始拉取 %d 只股票：%s", len(codes), codes)
    df = fetch_many(
        codes=codes,
        start_date=start,
        end_date=end,
        frequency=frequency,
        adjust=adjust,
        use_cache=not args.no_cache,
    )
    if df.empty:
        logger.warning("未拉取到任何数据")
        return

    out_path = DATA_DIR / "raw_all.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("已保存合并数据到 %s，共 %d 行", out_path, len(df))


def _load_raw() -> pd.DataFrame:
    raw_file = DATA_DIR / "raw_all.parquet"
    if not raw_file.exists():
        logger.error("未找到合并数据 %s，请先运行 fetch", raw_file)
        sys.exit(1)
    return pd.read_parquet(raw_file)


def cmd_fetch_min(args) -> None:
    """拉取部分股票的分钟线数据（存储到 cache/raw_min/{frequency}/）。"""
    from data.fetcher import fetch_min_many

    codes = parse_codes(args.codes) if args.codes else cfg.default_codes
    start = args.start or cfg.start_date
    end = args.end or cfg.end_date or date.today().isoformat()
    adjust = args.adjust or "2"
    frequency = args.frequency or "5"

    logger.info(
        "开始拉取 %d 只股票 %s 分钟线：%s ~ %s，adjust=%s",
        len(codes), frequency, start, end, adjust,
    )
    df = fetch_min_many(
        codes=codes,
        start_date=start,
        end_date=end,
        frequency=frequency,
        adjust=adjust,
        use_cache=not args.no_cache,
        refresh=args.refresh,
        sleep=args.sleep if args.sleep is not None else cfg.fetch_sleep,
    )
    if df.empty:
        logger.warning("未拉取到任何分钟线数据")
        return

    logger.info(
        "分钟线拉取完成：共 %d 行，已分片存储到 cache/raw_min/%s/",
        len(df), frequency,
    )


def cmd_fetch_external(args) -> None:
    """拉取外部数据：基本面事件（分红/财报/预告/快报）+ 宏观利率序列。"""
    from data.fetcher_external import (
        fetch_all_fundamental,
        fetch_fundamental_many,
        fetch_macro_all,
    )

    start = args.start or cfg.start_date
    end = args.end or cfg.end_date or date.today().isoformat()
    sleep = args.sleep if args.sleep is not None else cfg.fetch_sleep

    if args.macro:
        stats = fetch_macro_all(start_date=start, end_date=end)
        logger.info("宏观序列拉取完成：%s", stats)

    if args.all:
        logger.info(
            "开始全市场基本面拉取：区间 %s ~ %s，refresh=%s",
            start, end, args.refresh,
        )
        stats = fetch_all_fundamental(
            start_date=start,
            end_date=end,
            sleep=sleep,
            include_delisted=not args.no_delisted,
            refresh=args.refresh,
        )
        logger.info("全市场基本面拉取统计：%s", stats)
        return

    codes = parse_codes(args.codes) if args.codes else cfg.default_codes
    logger.info("开始拉取 %d 只股票基本面事件：%s", len(codes), codes)
    df = fetch_fundamental_many(
        codes=codes,
        start_date=start,
        end_date=end,
        use_cache=not args.no_cache,
        refresh=args.refresh,
        sleep=sleep,
    )
    logger.info(
        "基本面事件拉取完成：共 %d 条，已分片存储到 cache/external/fundamental/",
        len(df),
    )


def cmd_features(args) -> None:
    from data import store
    from features.builder import build_dataset, build_samples_to_store

    window = args.window if args.window is not None else cfg.window
    horizon = args.horizon if args.horizon is not None else cfg.horizon

    raw_codes = store.list_raw_codes()
    if raw_codes:
        # 分片模式：从 raw 分片流式构建 samples 分片（适合全市场）
        built = build_samples_to_store(
            window=window,
            horizon=horizon,
            overwrite=args.overwrite,
        )
        logger.info(
            "本次构建 %d 只；samples 分片共 %d 只，保存在 %s",
            len(built), len(store.list_sample_codes()), store.SAMPLES_DIR,
        )
        return

    # 兼容旧流程：从 raw_all.parquet 构建单文件 samples.parquet
    raw = _load_raw()
    sample = build_dataset(raw, window=window, horizon=horizon)
    out_path = DATA_DIR / "samples.parquet"
    sample.to_parquet(out_path, index=False)
    logger.info(
        "特征构建完成：%d 条样本，保存到 %s（特征列 %d 个）",
        len(sample), out_path, len([c for c in sample.columns if c not in ("date", "code", "close", "label_reg", "label_cls")]),
    )


def _load_samples() -> pd.DataFrame:
    from data import store

    # 优先读取 samples 分片目录（全量模式）
    if store.list_sample_codes():
        return store.load_all_samples()

    # 兼容旧版单文件
    sample_file = DATA_DIR / "samples.parquet"
    if not sample_file.exists():
        logger.error("未找到样本数据，请先运行 features（或 fetch 后 features）")
        sys.exit(1)
    return pd.read_parquet(sample_file)


def _model_mode(model: str) -> str:
    if model == "lstm":
        return "seq"
    return "flat"


def _effective_config(model_name: str, epochs: int) -> Config:
    """按 CLI 参数构造与训练一致的有效配置。"""
    return Config(
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        adjust=cfg.adjust,
        frequency=cfg.frequency,
        horizon=cfg.horizon,
        window=cfg.window,
        train_ratio=cfg.train_ratio,
        valid_ratio=cfg.valid_ratio,
        model=model_name,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        epochs=epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        seed=cfg.seed,
        device=cfg.device,
        default_codes=cfg.default_codes,
    )


def cmd_train(args) -> None:
    from features.dataset import build_bundle
    from models.train import train_model, evaluate_model, save_model

    sample = _load_samples()
    model_name = args.model or cfg.model

    effective_cfg = _effective_config(
        model_name, args.epochs if args.epochs else cfg.epochs
    )

    mode = _model_mode(model_name)
    logger.info("使用模型=%s, mode=%s", model_name, mode)

    bundle = build_bundle(
        sample,
        train_ratio=effective_cfg.train_ratio,
        valid_ratio=effective_cfg.valid_ratio,
        seq_len=effective_cfg.seq_len,
        mode=mode,
    )

    logger.info("开始训练...")
    model_info = train_model(bundle, effective_cfg)

    logger.info("评估测试集...")
    metrics = evaluate_model(model_info, bundle, effective_cfg)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    out_dir = DATA_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(model_info, str(out_dir / f"{model_name}.pt"))
    logger.info("训练完成，模型已保存到 %s", out_dir / f"{model_name}.pt")


def cmd_base_predict(args) -> None:
    """基座模型全量推理：输出 (date, code, base_pred)，date 为目标交易日。"""
    from features.dataset import build_bundle
    from models.train import predict_base_all

    model_name = args.model or cfg.model
    if model_name == "baseline":
        logger.error("base-predict 仅支持 NN 基座模型（lstm/mlp）")
        sys.exit(1)

    sample = _load_samples()
    effective_cfg = _effective_config(model_name, cfg.epochs)
    mode = _model_mode(model_name)

    bundle = build_bundle(
        sample,
        train_ratio=effective_cfg.train_ratio,
        valid_ratio=effective_cfg.valid_ratio,
        seq_len=effective_cfg.seq_len,
        mode=mode,
    )

    model_path = args.model_path or str(DATA_DIR / "models" / f"{model_name}.pt")
    if not Path(model_path).exists():
        logger.error("基座模型不存在: %s，请先运行 train", model_path)
        sys.exit(1)

    preds = predict_base_all(bundle, effective_cfg, model_path)
    out = DATA_DIR / "meta" / "base_preds.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    logger.info("基座预测已保存到 %s，共 %d 条", out, len(preds))


def cmd_min_features(args) -> None:
    """构造分钟级样本：分钟线序列 + 日线/压力位特征 + 基座预测。"""
    from data import store
    from features.min_builder import build_min_samples

    frequency = args.frequency or cfg.min_frequency
    base_path = args.base_preds or str(DATA_DIR / "meta" / "base_preds.parquet")
    if not Path(base_path).exists():
        logger.error("基座预测不存在: %s，请先运行 base-predict", base_path)
        sys.exit(1)
    base_preds = pd.read_parquet(base_path)

    built = build_min_samples(
        frequency=frequency,
        base_preds=base_preds,
        overwrite=args.overwrite,
    )
    logger.info(
        "本次构建 %d 只；分钟样本分片共 %d 只，保存在 cache/min_samples/%s/",
        len(built), len(store.list_min_sample_codes(frequency)), frequency,
    )


def cmd_min_train(args) -> None:
    """训练分钟级 IntradayLSTM。"""
    from data import store
    from features.min_dataset import build_min_bundle
    from models.train import save_min_model, train_min_model

    frequency = args.frequency or cfg.min_frequency
    sample = store.load_all_min_samples(frequency)
    effective_cfg = _effective_config(
        "intraday_lstm", args.epochs if args.epochs else cfg.epochs
    )

    bundle = build_min_bundle(
        sample,
        train_ratio=effective_cfg.train_ratio,
        valid_ratio=effective_cfg.valid_ratio,
    )

    logger.info("开始训练分钟模型（%s 分钟线）...", frequency)
    trainer, metrics = train_min_model(bundle, effective_cfg)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    out_dir = DATA_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(out_dir / f"intraday_lstm_{frequency}.pt")
    save_min_model(trainer, bundle, model_path)
    logger.info("分钟模型已保存到 %s", model_path)


def cmd_min_predict(args) -> None:
    """盘中预测：拉最新分钟线 → 10:00 截面 → 输出当日收盘预测。"""
    import numpy as np

    from data.fetcher import fetch_minutes, fetch_stock
    from features.min_builder import build_predict_input

    code = args.code
    frequency = args.frequency or cfg.min_frequency

    # 1. 更新数据（分钟线 + 日线）
    end = date.today().isoformat()
    fetch_stock(code, start_date=cfg.start_date, end_date=end, refresh=True)
    fetch_minutes(code, start_date=cfg.start_date, end_date=end,
                  frequency=frequency, refresh=True)

    # 2. 基座预测
    base_pred = None
    base_path = DATA_DIR / "meta" / "base_preds.parquet"
    if base_path.exists():
        base_preds = pd.read_parquet(base_path)
        hit = base_preds[
            (base_preds["code"] == code)
        ].sort_values("date")
        if not hit.empty:
            base_pred = float(hit["base_pred"].iloc[-1])
    if base_pred is None:
        logger.warning("未找到基座预测（%s），base_pred 以 0 填充", code)

    # 3. 构造 10:00 截面输入
    inp = build_predict_input(code, frequency, base_pred=base_pred)
    if inp is None:
        sys.exit(1)

    # 4. 加载模型并推理
    try:
        import torch
        from models.nn import IntradayLSTM
    except ImportError:
        logger.error("未安装 torch，无法加载分钟模型")
        sys.exit(1)

    model_path = DATA_DIR / "models" / f"intraday_lstm_{frequency}.pt"
    if not model_path.exists():
        logger.error("分钟模型不存在: %s，请先运行 min-train", model_path)
        sys.exit(1)
    ckpt = torch.load(model_path, map_location="cpu")

    net = IntradayLSTM(
        seq_input_dim=ckpt["seq_input_dim"],
        static_dim=len(ckpt["static_cols"]),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    static_vals = np.asarray(
        [inp["static"].get(c, 0.0) for c in ckpt["static_cols"]],
        dtype=np.float32,
    ).reshape(1, -1)
    x_seq = torch.tensor(inp["seq"], dtype=torch.float32)
    x_static = torch.tensor(static_vals, dtype=torch.float32)
    with torch.no_grad():
        pred_reg, pred_cls_logit = net(x_seq, x_static)
    pred_rest = float(pred_reg.squeeze(0).item())
    prob_up = float(torch.sigmoid(pred_cls_logit).squeeze(0).item())

    pred_close = inp["close10"] * (1.0 + pred_rest / 100.0)

    print(json.dumps({
        "code": code,
        "date": str(inp["date"].date()),
        "open": round(inp["open_day"], 2),
        "price_at_1000": round(inp["close10"], 2),
        "predicted_change_pct_from_1000": round(pred_rest, 3),
        "predicted_close": round(pred_close, 2),
        "prob_close_up_from_1000": round(prob_up, 3),
        "base_pred_pct": base_pred,
    }, indent=2, ensure_ascii=False))


def cmd_predict(args) -> None:
    logger.warning(
        "predict 子命令为占位实现：请先运行 train 生成模型，"
        "再基于最新特征调用对应模型推理。完整推理示例见 README。"
    )
    # 简单演示：读取样本并显示最近一条
    sample = _load_samples()
    code = args.code
    if code:
        sub = sample[sample["code"] == code]
    else:
        sub = sample
    if sub.empty:
        logger.error("未找到代码 %s 的样本", code)
        return
    latest = sub.sort_values("date").tail(3)
    cols = ["date", "code", "close", "label_reg", "label_cls"]
    print(latest[cols].to_string(index=False))


# ---- 参数解析 ----


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A 股涨跌幅预测系统")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")

    sub = parser.add_subparsers(dest="command", required=True)

    # fetch
    p_fetch = sub.add_parser("fetch", help="拉取历史数据")
    p_fetch.add_argument("--all", action="store_true",
                         help="拉取全市场股票（分片存储到 cache/raw/，支持断点续传）")
    p_fetch.add_argument("--refresh", action="store_true",
                         help="增量更新：拉取缓存最后日期之后的数据（需配合 --all）")
    p_fetch.add_argument("--codes", type=str, help="逗号分隔的股票代码，如 sh.600000,sz.000001")
    p_fetch.add_argument("--start", type=str, help="开始日期 YYYY-MM-DD")
    p_fetch.add_argument("--end", type=str, help="结束日期 YYYY-MM-DD（默认今天）")
    p_fetch.add_argument("--adjust", type=str, help="复权：1后复权 2前复权 3不复权")
    p_fetch.add_argument("--frequency", type=str, help="周期 d/w/m")
    p_fetch.add_argument("--sleep", type=float, help="每只股票之间的间隔秒数")
    p_fetch.add_argument("--no-delisted", action="store_true", help="全量拉取时排除退市股")
    p_fetch.add_argument("--no-cache", action="store_true", help="忽略缓存强制拉取")
    p_fetch.set_defaults(func=cmd_fetch)

    # fetch-min（分钟线，仅部分股票）
    p_fmin = sub.add_parser("fetch-min", help="拉取部分股票的分钟线数据")
    p_fmin.add_argument("--codes", type=str,
                        help="逗号分隔的股票代码，如 sh.600000,sz.000001（默认 cfg.default_codes）")
    p_fmin.add_argument("--frequency", type=str, choices=["5", "15", "30", "60"],
                        help="分钟线频率（默认 5）")
    p_fmin.add_argument("--start", type=str, help="开始日期 YYYY-MM-DD")
    p_fmin.add_argument("--end", type=str, help="结束日期 YYYY-MM-DD（默认今天）")
    p_fmin.add_argument("--adjust", type=str, help="复权：1后复权 2前复权（默认） 3不复权")
    p_fmin.add_argument("--refresh", action="store_true", help="增量更新到最新交易日")
    p_fmin.add_argument("--sleep", type=float, help="每只股票之间的间隔秒数")
    p_fmin.add_argument("--no-cache", action="store_true", help="忽略缓存强制拉取")
    p_fmin.set_defaults(func=cmd_fetch_min)

    # fetch-external（基本面事件 + 宏观利率序列）
    p_fext = sub.add_parser("fetch-external", help="拉取外部数据(分红/财报/业绩预告/宏观利率)")
    p_fext.add_argument("--codes", type=str,
                        help="逗号分隔的股票代码（默认 cfg.default_codes）")
    p_fext.add_argument("--all", action="store_true",
                        help="全市场股票基本面事件（分片存储，支持断点续传）")
    p_fext.add_argument("--macro", action="store_true",
                        help="拉取宏观利率序列（存/贷款利率、准备金率、货币供应量）")
    p_fext.add_argument("--start", type=str, help="开始日期 YYYY-MM-DD")
    p_fext.add_argument("--end", type=str, help="结束日期 YYYY-MM-DD（默认今天）")
    p_fext.add_argument("--refresh", action="store_true",
                        help="重拉已缓存股票的基本面事件")
    p_fext.add_argument("--sleep", type=float, help="每只股票之间的间隔秒数")
    p_fext.add_argument("--no-delisted", action="store_true",
                        help="全量拉取时排除退市股")
    p_fext.add_argument("--no-cache", action="store_true", help="忽略缓存强制拉取")
    p_fext.set_defaults(func=cmd_fetch_external)

    # features
    p_feat = sub.add_parser("features", help="构建特征与标签")
    p_feat.add_argument("--window", type=int, help="回看窗口")
    p_feat.add_argument("--horizon", type=int, help="预测周期")
    p_feat.add_argument("--overwrite", action="store_true", help="重建已存在的 samples 分片")
    p_feat.set_defaults(func=cmd_features)

    # base-predict（基座全量推理，供分钟模型级联）
    p_base = sub.add_parser("base-predict", help="基座模型全量推理，生成 base_preds")
    p_base.add_argument("--model", type=str, choices=["lstm", "mlp"], help="基座模型类型")
    p_base.add_argument("--model-path", type=str, help="基座模型文件路径")
    p_base.set_defaults(func=cmd_base_predict)

    # min-features（分钟样本构造）
    p_minf = sub.add_parser("min-features", help="构造分钟级预测样本")
    p_minf.add_argument("--frequency", type=str, choices=["5", "15", "30", "60"],
                        help="分钟线频率（默认 5）")
    p_minf.add_argument("--base-preds", type=str, help="基座预测文件路径")
    p_minf.add_argument("--overwrite", action="store_true", help="重建已存在的样本分片")
    p_minf.set_defaults(func=cmd_min_features)

    # min-train（分钟模型训练）
    p_mint = sub.add_parser("min-train", help="训练分钟级模型 IntradayLSTM")
    p_mint.add_argument("--frequency", type=str, choices=["5", "15", "30", "60"],
                        help="分钟线频率（默认 5）")
    p_mint.add_argument("--epochs", type=int, help="训练轮数")
    p_mint.set_defaults(func=cmd_min_train)

    # min-predict（盘中预测）
    p_minp = sub.add_parser("min-predict", help="盘中预测当日收盘价")
    p_minp.add_argument("--code", type=str, required=True, help="股票代码，如 sh.600000")
    p_minp.add_argument("--frequency", type=str, choices=["5", "15", "30", "60"],
                        help="分钟线频率（默认 5）")
    p_minp.set_defaults(func=cmd_min_predict)

    # train
    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--model", type=str, choices=["lstm", "mlp", "baseline"], help="模型类型")
    p_train.add_argument("--epochs", type=int, help="训练轮数（NN）")
    p_train.set_defaults(func=cmd_train)

    # predict
    p_pred = sub.add_parser("predict", help="预测（占位演示）")
    p_pred.add_argument("--code", type=str, help="股票代码")
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()