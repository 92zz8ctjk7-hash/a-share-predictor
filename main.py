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
    from features.min_builder import build_min_samples, build_roll_samples

    frequency = args.frequency or cfg.min_frequency
    base_path = args.base_preds or str(DATA_DIR / "meta" / "base_preds.parquet")
    if not Path(base_path).exists():
        logger.error("基座预测不存在: %s，请先运行 base-predict", base_path)
        sys.exit(1)
    base_preds = pd.read_parquet(base_path)

    if args.rolling:
        # 滚动窗口样本（当前 30 分钟 → 预测下一个 30 分钟），存 min_samples_roll/
        built = []
        for code in store.list_min_codes(frequency):
            sample = build_roll_samples(code, frequency, base_preds)
            if sample is not None and not sample.empty:
                store.save_roll_samples(sample, code, frequency)
                built.append(code)
                logger.info("%s 滚动样本构建完成：%d 条", code, len(sample))
        logger.info(
            "本次构建滚动样本 %d 只，保存在 cache/min_samples_roll/%s/",
            len(built), frequency,
        )
        return

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
    from serving.predict import predict_intraday

    signal = predict_intraday(
        args.code,
        frequency=args.frequency or cfg.min_frequency,
        realtime=not args.no_realtime,
    )
    if signal is None:
        sys.exit(1)
    print(json.dumps(signal, indent=2, ensure_ascii=False))


def cmd_serve(args) -> None:
    """定时服务任务：预测 → 信号持久化 → 企业微信推送（手动或 launchd 触发）。"""
    from serving.scheduler import run_daily_job

    codes = parse_codes(args.codes) if args.codes else ["sz.000100"]
    results = run_daily_job(codes, frequency=args.frequency, dry_run=args.dry_run)
    logger.info("serve 完成：%d 个信号已生成", len(results))


def cmd_install_scheduler(args) -> None:
    """生成并安装 launchd 定时任务（工作日盘中自动出信号）。"""
    from serving.scheduler import install_scheduler

    codes = parse_codes(args.codes) if args.codes else ["sz.000100"]
    install_scheduler(codes, frequency=args.frequency, load=args.load)


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


def cmd_backtest(args) -> None:
    """多窗口网格交易回测：窗口前段训练、末段样本外回测。"""
    from backtest.run import run_backtest

    codes = parse_codes(args.codes) if args.codes else cfg.default_codes
    windows = (
        [w.strip() for w in args.windows.split(",") if w.strip()]
        if args.windows else cfg.bt_windows
    )
    report = run_backtest(
        codes=codes,
        capital=args.capital if args.capital else cfg.bt_capital,
        windows=windows,
        model=args.model or "baseline",
        grid_n=args.grid_n if args.grid_n is not None else cfg.bt_grid_n,
        range_pct=args.range_pct if args.range_pct is not None else cfg.bt_range_pct,
        gate_on=not args.no_gate,
        gate_threshold=args.gate_threshold
        if args.gate_threshold is not None else cfg.bt_gate_threshold,
        commission=args.commission if args.commission is not None else cfg.bt_commission,
        stamp_tax=args.stamp_tax if args.stamp_tax is not None else cfg.bt_stamp_tax,
        slippage=args.slippage if args.slippage is not None else cfg.bt_slippage,
        train_ratio=args.train_ratio if args.train_ratio is not None else cfg.bt_train_ratio,
        epochs=args.epochs,
        save_every=args.save_every if args.save_every is not None else cfg.bt_save_every,
        keep_curves=not args.no_curves,
        signal_source=args.signal_source or "base",
        min_frequency=args.min_frequency or "5",
        pretrained=args.pretrained,
    )
    if report.empty:
        return
    show_cols = [
        "code", "window", "capital", "final_value", "total_return_pct",
        "annual_return_pct", "max_drawdown_pct", "sharpe", "win_rate",
        "n_trades", "buy_hold_return_pct", "excess_pct",
    ]
    show_cols = [c for c in show_cols if c in report.columns]
    print("\n===== 回测报告（网格策略 vs 买入持有）=====")
    print(report[show_cols].to_string(index=False))


def cmd_factor_ic(args) -> None:
    """因子 IC 分析：验证「当天因子 → 未来收益」的预测有效性。"""
    from backtest.factor_eval import factor_ic_report
    from backtest.run import ensure_data
    from features.builder import build_dataset

    codes = parse_codes(args.codes) if args.codes else cfg.default_codes
    frames = []
    for code in codes:
        bars = ensure_data(code)
        if bars is None or bars.empty:
            logger.warning("无 %s 数据，跳过", code)
            continue
        sample = build_dataset(bars, window=cfg.window, horizon=args.horizon or cfg.horizon)
        if not sample.empty:
            frames.append(sample)
    if not frames:
        logger.error("无可用样本")
        sys.exit(1)

    sample_all = pd.concat(frames, ignore_index=True)
    report = factor_ic_report(sample_all, rolling=args.rolling or 60)
    if report.empty:
        return
    print("\n===== 因子 IC 报告（按 |IC 均值| 降序）=====")
    print(report.to_string(index=False))


def cmd_incremental_update(args) -> None:
    """一键增量更新：刷新数据→重建样本→滚动窗口重训基座与分钟模型。"""
    import json

    from models.incremental import run_incremental_update

    codes = parse_codes(args.codes) if args.codes else ["sz.000100"]
    summary = run_incremental_update(
        codes=codes,
        base_window=args.base_window or "2y",
        min_window=args.min_window or "12m",
        base_model=args.base_model or "lstm",
        base_epochs=args.base_epochs if args.base_epochs is not None else 10,
        min_epochs=args.min_epochs if args.min_epochs is not None else 30,
        resume=args.resume,
        frequency=args.frequency or "5",
    )
    print("\n===== 增量更新完成 =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("模型已更新，可用 min-predict 基于最新模型实时预测")


def cmd_plot_equity(args) -> None:
    """绘制回测资金曲线（需先运行过 backtest）。"""
    from backtest.plot import plot_equity

    out = plot_equity(args.code, args.window)
    print(out)


def cmd_rl_backtest(args) -> None:
    """Bandit 门控 walk-forward 四模式对比回测。"""
    from rl_gate.backtest import run_rl_backtest

    codes = parse_codes(args.codes) if args.codes else ["sz.000100"]
    windows = (
        [w.strip() for w in args.windows.split(",") if w.strip()]
        if args.windows else cfg.bt_windows
    )
    if args.walk:
        from rl_gate.backtest import run_rl_walk

        report = run_rl_walk(
            codes=codes,
            n_segments=args.segments if args.segments else 4,
            capital=args.capital if args.capital else cfg.bt_capital,
            model=args.model or "lstm",
            epochs=args.epochs if args.epochs is not None else 10,
        )
        if report.empty:
            return
        seg_cols = ["code", "segment", "mode", "range", "regime", "gate_auc",
                    "total_return_pct", "max_drawdown_pct", "sharpe", "win_rate",
                    "n_trades", "buy_hold_return_pct", "excess_pct"]
        agg_cols = ["code", "segment", "mode", "range", "final_value",
                    "total_return_pct", "annual_return_pct"]
        det = report[report["segment"] != "ALL"]
        agg = report[report["segment"] == "ALL"]
        print("\n===== 分段对比（每段严格样本外）=====")
        print(det[[c for c in seg_cols if c in det.columns]].to_string(index=False))
        print("\n===== 全周期汇总（资金跨段连续）=====")
        print(agg[[c for c in agg_cols if c in agg.columns]].to_string(index=False))
        return
    report = run_rl_backtest(
        codes=codes,
        windows=windows,
        capital=args.capital if args.capital else cfg.bt_capital,
        model=args.model or "lstm",
        epochs=args.epochs if args.epochs is not None else 10,
    )
    if report.empty:
        return
    show_cols = [
        "code", "window", "mode", "gate_auc", "best_theta",
        "total_return_pct", "max_drawdown_pct", "sharpe", "win_rate",
        "n_trades", "buy_hold_return_pct", "excess_pct",
    ]
    show_cols = [c for c in show_cols if c in report.columns]
    print("\n===== RL 门控四模式对比（严格样本外）=====")
    print(report[show_cols].to_string(index=False))


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
    p_minf.add_argument("--rolling", action="store_true",
                        help="构建滚动窗口样本（当前 30 分钟 → 预测下一个 30 分钟）")
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
    p_minp.add_argument("--no-realtime", action="store_true",
                        help="禁用 akshare 实时分钟源，改用 baostock 分钟线")
    p_minp.set_defaults(func=cmd_min_predict)

    # serve（定时服务任务：预测→持久化→推送）
    p_serve = sub.add_parser("serve", help="执行一次服务任务（手动或 launchd 触发）")
    p_serve.add_argument("--codes", type=str, help="股票代码，逗号分隔（默认 sz.000100）")
    p_serve.add_argument("--frequency", type=str, choices=["5", "15", "30", "60"],
                         help="分钟线频率（默认 5）")
    p_serve.add_argument("--dry-run", action="store_true",
                         help="只持久化并打印信号，不推送企业微信")
    p_serve.set_defaults(func=cmd_serve)

    # install-scheduler（生成并安装 launchd 定时任务）
    p_is = sub.add_parser("install-scheduler", help="生成并安装 launchd 定时任务")
    p_is.add_argument("--codes", type=str, help="股票代码，逗号分隔（默认 sz.000100）")
    p_is.add_argument("--frequency", type=str, choices=["5", "15", "30", "60"],
                      help="分钟线频率（默认 5）")
    p_is.add_argument("--load", action="store_true",
                      help="安装后立即 launchctl load 启用")
    p_is.set_defaults(func=cmd_install_scheduler)

    # backtest（多窗口网格交易回测）
    p_bt = sub.add_parser("backtest", help="多窗口网格交易回测（含模型训练与 checkpoint）")
    p_bt.add_argument("--codes", type=str,
                      help="逗号分隔的股票代码，如 sh.600000,sz.000100")
    p_bt.add_argument("--capital", type=float, help="初始资金（默认 100000）")
    p_bt.add_argument("--windows", type=str, help="回测窗口，逗号分隔，如 5y,3y,2y,1y")
    p_bt.add_argument("--model", type=str, choices=["baseline", "mlp", "lstm"],
                      help="预测模型（默认 baseline）")
    p_bt.add_argument("--grid-n", type=int, help="网格数量（默认 10）")
    p_bt.add_argument("--range-pct", type=float, help="网格上下界幅度（默认 0.2 即 ±20%%）")
    p_bt.add_argument("--no-gate", action="store_true", help="关闭模型信号门控（纯网格）")
    p_bt.add_argument("--gate-threshold", type=float, help="上涨概率门控阈值（默认 0.5）")
    p_bt.add_argument("--commission", type=float, help="佣金率（默认万 2.5）")
    p_bt.add_argument("--stamp-tax", type=float, help="印花税，仅卖出（默认 0.05%%）")
    p_bt.add_argument("--slippage", type=float, help="滑点比例（默认 0.1%%）")
    p_bt.add_argument("--train-ratio", type=float, help="窗口内训练段占比（默认 0.8）")
    p_bt.add_argument("--epochs", type=int, help="NN 训练轮数")
    p_bt.add_argument("--save-every", type=int, help="checkpoint 保存间隔（默认 10 epoch）")
    p_bt.add_argument("--signal-source", type=str, choices=["base", "cascade"],
                      help="信号来源：base=日线模型（默认）；cascade=分钟模型收盘撮合+基座回退")
    p_bt.add_argument("--min-frequency", type=str, choices=["5", "15", "30", "60"],
                      help="cascade 模式的分钟线频率（默认 5）")
    p_bt.add_argument("--no-curves", action="store_true", help="不保存每日资金曲线 CSV")
    p_bt.add_argument("--pretrained", action="store_true",
                      help="直接加载已有 checkpoint（cache/models/{model}.pt）回测，不重新训练")
    p_bt.set_defaults(func=cmd_backtest)

    # factor-ic（因子有效性分析）
    p_ic = sub.add_parser("factor-ic", help="因子 IC 分析（IC/ICIR/分层收益）")
    p_ic.add_argument("--codes", type=str,
                      help="逗号分隔的股票代码（>=2 只时用截面 IC，单只用滚动 IC）")
    p_ic.add_argument("--horizon", type=int, help="未来收益周期（默认 cfg.horizon）")
    p_ic.add_argument("--rolling", type=int, help="单股模式滚动窗口（默认 60 日）")
    p_ic.set_defaults(func=cmd_factor_ic)

    # incremental-update（一键增量训练）
    p_inc = sub.add_parser("incremental-update", help="一键增量更新（刷新数据+滚动窗口重训基座与分钟模型）")
    p_inc.add_argument("--codes", type=str, help="分钟模型标的，逗号分隔（默认 sz.000100）")
    p_inc.add_argument("--base-window", type=str, help="基座训练窗口，如 2y/12m/90d（默认 2y）")
    p_inc.add_argument("--min-window", type=str, help="分钟训练窗口，如 12m/90d（默认 12m）")
    p_inc.add_argument("--base-model", type=str, choices=["lstm", "mlp"], help="基座模型（默认 lstm）")
    p_inc.add_argument("--base-epochs", type=int, help="基座训练轮数（默认 10）")
    p_inc.add_argument("--min-epochs", type=int, help="分钟模型训练轮数（默认 30）")
    p_inc.add_argument("--resume", action="store_true", help="加载已有权重 warm-start 继续训练")
    p_inc.add_argument("--frequency", type=str, choices=["5", "15", "30", "60"], help="分钟线频率（默认 5）")
    p_inc.set_defaults(func=cmd_incremental_update)

    # plot-equity（回测资金曲线绘图）
    p_plot = sub.add_parser("plot-equity", help="绘制回测资金曲线（需先运行过 backtest）")
    p_plot.add_argument("--code", type=str, required=True, help="股票代码，如 sz.000100")
    p_plot.add_argument("--window", type=str, required=True,
                        help="回测窗口，如 5y/3y/2y/1y 或 min_5")
    p_plot.set_defaults(func=cmd_plot_equity)

    # rl-backtest（Bandit 门控 walk-forward 对比）
    p_rl = sub.add_parser("rl-backtest", help="Bandit 门控四模式对比回测（严格样本外）")
    p_rl.add_argument("--codes", type=str, help="回测标的，逗号分隔（默认 sz.000100）")
    p_rl.add_argument("--windows", type=str, help="回测窗口，逗号分隔（默认全部）")
    p_rl.add_argument("--capital", type=float, help="初始资金（默认 100000）")
    p_rl.add_argument("--model", type=str, choices=["lstm", "mlp"],
                      help="日线信号模型（默认 lstm）")
    p_rl.add_argument("--epochs", type=int, help="日线模型训练轮数（默认 10）")
    p_rl.add_argument("--walk", action="store_true",
                      help="全周期滚动 walk-forward（切多段覆盖牛熊震荡，资金跨段连续）")
    p_rl.add_argument("--segments", type=int, help="walk 模式测试段数（默认 4）")
    p_rl.set_defaults(func=cmd_rl_backtest)

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