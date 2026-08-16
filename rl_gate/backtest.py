"""walk-forward 四模式回测：验证 Bandit 门控的样本外真实价值。

对每个 (code, window)：
    cutoff = 窗口起点 + 75% 时间点（训练段/测试段边界）
    1. gate：沪深 300 机会样本（cutoff 前）训练逻辑回归门控
    2. 日线 LSTM：训练段重训（_train_and_predict），对测试段出 prob_up 信号
    3. 四模式回测测试段（同一网格参数）：
       - none    纯网格（无门控）
       - fixed   固定阈值 0.5 门控（现状）
       - rolling 方案 A：训练段网格搜索 θ 选夏普最优，应用到测试段
       - gate    Bandit 门控（市场状态 → 放行买入）
    4. 对比报告 + 四曲线对比图

严格口径：测试段数据不参与 gate 与 LSTM 的任何训练。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import DATA_DIR, Config, cfg

logger = logging.getLogger(__name__)

# 训练段占窗口比例（其余为测试段）
TRAIN_RATIO = 0.75

MODES = ["none", "fixed", "rolling", "gate"]

# 方案 A 的阈值搜索网格
THETA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _load_base_codes() -> List[str]:
    p = DATA_DIR / "meta" / "hs300_codes.txt"
    if p.exists():
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    from data import store

    return store.list_raw_codes()


def _search_theta(
    bars_train, signals_train, capital, cost, grid_n, range_pct
) -> float:
    """方案 A：在训练段网格搜索门控阈值，返回夏普最优 θ。"""
    from backtest.engine import run_engine
    from backtest.strategy import GridStrategy

    best_theta, best_sharpe = 0.5, -1e18
    for theta in THETA_GRID:
        strat = GridStrategy(
            grid_n=grid_n, range_pct=range_pct,
            gate_on=True, gate_threshold=theta,
        )
        res = run_engine(bars_train, signals_train, strat, capital=capital, cost=cost)
        sharpe = res.stats.get("sharpe", -1e18)
        if sharpe > best_sharpe:
            best_sharpe, best_theta = sharpe, theta
    logger.info("方案 A 训练段搜索完成：最优 θ=%.1f（夏普 %.2f）", best_theta, best_sharpe)
    return best_theta


def _plot_comparison(code: str, window: str, curves: Dict[str, pd.DataFrame]) -> None:
    """四模式资金曲线 + 买入持有画在一张图。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过绘图")
        return

    plt.rcParams["font.family"] = ["Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    colors = {"none": "#7f7f7f", "fixed": "#ff7f0e", "rolling": "#2ca02c",
              "gate": "#d62728", "dqn": "#9467bd", "hybrid": "#17becf"}
    labels = {"none": "纯网格", "fixed": "固定阈值0.5", "rolling": "滚动最优θ",
              "gate": "Bandit门控", "dqn": "DQN门控(RL)", "hybrid": "混合门控(regime切换)"}

    fig, ax = plt.subplots(figsize=(13, 7))
    ref = None
    for mode in MODES:
        df = curves.get(mode)
        if df is None or df.empty:
            continue
        ref = df
        ret = (df["total"] / df["total"].iloc[0] - 1) * 100
        ax.plot(df["date"], ret, color=colors[mode], lw=1.6, label=labels[mode])

    if ref is not None:
        bh = (ref["close"] / ref["close"].iloc[0] - 1) * 100
        ax.plot(ref["date"], bh, color="#bbbbbb", lw=1.0, ls=":", label="买入持有")

    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_title(f"{code} · 窗口 {window} · 四模式门控对比（严格样本外，测试段收益 %）")
    ax.set_ylabel("收益（%）")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = DATA_DIR / "meta" / f"equity_rl_{code}_{window}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("四模式对比图已保存到 %s", out)


def run_rl_backtest(
    codes: List[str],
    windows: Optional[List[str]] = None,
    capital: Optional[float] = None,
    model: str = "lstm",
    epochs: int = 10,
    grid_n: int = 10,
    range_pct: float = 0.20,
) -> pd.DataFrame:
    """walk-forward 四模式回测主入口，返回对比报告 DataFrame。"""
    from backtest.engine import CostConfig, run_engine
    from backtest.run import _train_and_predict, _window_start
    from backtest.strategy import GridStrategy
    from data import store
    from features.builder import build_dataset
    from rl_gate.gate import make_buy_gate, train_gate
    from rl_gate.opportunity import build_day_features, build_opportunities, get_features

    windows = windows or cfg.bt_windows
    capital = capital if capital is not None else cfg.bt_capital
    cost = CostConfig()
    base_codes = _load_base_codes()
    meta_dir = DATA_DIR / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 沪深 300 机会样本全量构造一次，各窗口按 cutoff 过滤（防泄漏）
    logger.info("构造沪深 300 机会样本（一次性）...")
    opp_all = build_opportunities(base_codes)
    if opp_all.empty:
        logger.error("机会样本为空，无法继续")
        return pd.DataFrame()
    feats_list = get_features(True)

    rows: List[Dict] = []
    for code in codes:
        bars = store.load_bars(code)
        if bars is None or bars.empty:
            logger.error("无 %s 行情数据，跳过", code)
            continue
        sample = build_dataset(bars, window=cfg.window, horizon=cfg.horizon)
        if sample.empty:
            continue

        max_date = sample["date"].max()
        for window in windows:
            win_start = _window_start(max_date, window)
            sub = sample[sample["date"] >= win_start]
            if len(sub) < 120:
                logger.warning("%s %s 窗口样本不足，跳过", code, window)
                continue

            cutoff = sub["date"].quantile(TRAIN_RATIO)
            test_start = cutoff + pd.Timedelta(days=1)
            logger.info(
                "=== %s | 窗口 %s | 训练段 ≤ %s | 测试段 %s 起（严格样本外）===",
                code, window, cutoff.date(), test_start.date(),
            )

            # 1. gate 训练（沪深 300 联合，仅 cutoff 前）
            gate_bundle = train_gate(opp_all[opp_all["date"] < cutoff], feats_list)

            # 2. 日线 LSTM 训练段重训 → 全窗口信号
            ckpt_dir = DATA_DIR / "models" / "checkpoints" / f"rl_{code}_{model}_{window}"
            signals = _train_and_predict(
                sub, model, ckpt_dir, epochs, save_every=10 ** 6, train_ratio=TRAIN_RATIO
            )

            # 3. 切分训练/测试
            bars_train = bars[(bars["date"] >= win_start) & (bars["date"] < test_start)]
            bars_test = bars[bars["date"] >= test_start].reset_index(drop=True)
            sig_train = signals[(signals["date"] >= win_start) & (signals["date"] < test_start)]
            sig_test = signals[signals["date"] >= test_start]

            # 4. 方案 A：训练段搜索最优 θ
            best_theta = _search_theta(
                bars_train, sig_train, capital, cost, grid_n, range_pct
            )

            # 5. 四模式回测测试段
            results = {}
            results["none"] = run_engine(
                bars_test, None,
                GridStrategy(grid_n=grid_n, range_pct=range_pct, gate_on=False),
                capital=capital, cost=cost,
            )
            results["fixed"] = run_engine(
                bars_test, sig_test,
                GridStrategy(grid_n=grid_n, range_pct=range_pct,
                             gate_on=True, gate_threshold=0.5),
                capital=capital, cost=cost,
            )
            results["rolling"] = run_engine(
                bars_test, sig_test,
                GridStrategy(grid_n=grid_n, range_pct=range_pct,
                             gate_on=True, gate_threshold=best_theta),
                capital=capital, cost=cost,
            )
            feats_test = build_day_features(bars)
            feats_test = feats_test[feats_test["date"] >= test_start]
            buy_gate = make_buy_gate(gate_bundle, feats_test)
            results["gate"] = run_engine(
                bars_test, None,
                GridStrategy(grid_n=grid_n, range_pct=range_pct,
                             gate_on=False, buy_gate=buy_gate),
                capital=capital, cost=cost,
            )

            curves = {}
            for mode, res in results.items():
                curves[mode] = res.equity_curve
                row = {
                    "code": code, "window": window, "mode": mode, "capital": capital,
                    "gate_auc": round(gate_bundle["auc"], 4),
                    "best_theta": best_theta,
                    **res.stats,
                }
                rows.append(row)
                cur_path = meta_dir / f"equity_rl_{code}_{window}_{mode}.csv"
                if res.equity_curve is not None:
                    res.equity_curve.to_csv(cur_path, index=False)

            _plot_comparison(code, window, curves)

    report = pd.DataFrame(rows)
    if report.empty:
        logger.warning("无任何回测结果")
        return report
    out = meta_dir / "rl_gate_report.csv"
    report.to_csv(out, index=False)
    logger.info("RL 门控对比报告已保存到 %s（%d 行）", out, len(report))
    return report


# ---- 全周期滚动 walk-forward：覆盖牛/熊/震荡多 regime ----


def _train_and_predict_segment(
    sample: pd.DataFrame,
    cutoff,
    seg_end,
    model_name: str,
    ckpt_dir,
    epochs: int,
) -> pd.DataFrame:
    """用 cutoff 前全部样本训练，预测 [cutoff, seg_end] 段信号。"""
    from features.dataset import _feature_cols_of, build_bundle
    from models.train import load_nn_checkpoint, nn_predict, train_model

    train_part = sample[sample["date"] < cutoff].reset_index(drop=True)
    test_part = sample[
        (sample["date"] >= cutoff) & (sample["date"] <= seg_end)
    ].reset_index(drop=True)

    mode = "seq" if model_name == "lstm" else "flat"
    seq_len = cfg.seq_len
    bundle = build_bundle(
        train_part, train_ratio=0.85, valid_ratio=0.15,
        seq_len=seq_len, mode=mode,
    )
    eff_cfg = Config(
        start_date=cfg.start_date, end_date=cfg.end_date,
        horizon=cfg.horizon, window=cfg.window,
        model=model_name,
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        seq_len=seq_len, dropout=cfg.dropout,
        epochs=epochs, batch_size=cfg.batch_size, lr=cfg.lr,
        seed=cfg.seed, device=cfg.device,
    )
    train_model(bundle, eff_cfg, ckpt_dir=str(ckpt_dir), save_every=10 ** 6)

    net, meta = load_nn_checkpoint(str(ckpt_dir / "best.pt"))
    cols = meta["feature_names"]
    if mode == "seq":
        context = train_part[cols].tail(seq_len - 1)
        seq_src = pd.concat([context, test_part[cols]], ignore_index=True)
        X = seq_src.to_numpy(dtype=np.float32)
        pred_reg, prob_up = nn_predict(net, X, mode="seq", seq_len=seq_len)
    else:
        X = test_part[cols].to_numpy(dtype=np.float32)
        pred_reg, prob_up = nn_predict(net, X, mode="flat")
    return pd.DataFrame(
        {"date": test_part["date"].to_numpy(), "pred_reg": pred_reg,
         "prob_up": prob_up, "timing": "next_open"}
    )


def _build_envs(
    codes, before_date, grid_n, range_pct, capital, cost,
    include_market, min_days: int = 250,
) -> Dict:
    """构建多股票网格环境（仅用 before_date 之前的数据，供 DQN 训练）。"""
    from data import store
    from rl_gate.env import GridBacktestEnv
    from rl_gate.opportunity import build_day_features

    envs = {}
    for code in codes:
        bars = store.load_bars(code)
        if bars is None or bars.empty:
            continue
        bars_pre = bars[bars["date"] < before_date]
        if len(bars_pre) < min_days:
            continue
        feats = build_day_features(bars_pre, with_market=include_market)
        envs[code] = GridBacktestEnv(
            bars_pre, feats, grid_n=grid_n, range_pct=range_pct,
            capital=capital, cost=cost,
        )
    return envs


def _run_hybrid_episode(
    env, dqn_agent, logistic_buy_gate, regime_by_date, trend_th: float
) -> Tuple[pd.DataFrame, object]:
    """混合门控：趋势日用 logistic 决策，震荡日用 DQN 决策，同一 env 执行。

    结合两者优势：logistic 在趋势市稳定，DQN 在震荡市风险调整后最优。
    regime 由 |idx_ret_20d| > trend_th 判定（True=趋势市）。
    """
    s = env.reset()
    done = False
    rows: List[Dict] = []
    while not done:
        day_t = env.bars["date"].iloc[env.t]
        is_trend = regime_by_date.get(day_t, False)
        if is_trend:  # 趋势市 → logistic
            a = 1 if logistic_buy_gate(day_t) else 0
            pol = "logistic"
        else:  # 震荡市 → DQN
            a = dqn_agent.act(s, explore=False)
            pol = "dqn"
        s, r, done, info = env.step(a)
        info["policy"] = pol
        rows.append(info)
    return pd.DataFrame(rows), env


def _plot_walk_comparison(
    code: str, seg_curves: List[Dict], seg_info: List[Dict], out_suffix: str = ""
) -> None:
    """每段一个子图，四模式曲线 + 买入持有，标注 regime。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.rcParams["font.family"] = ["Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"none": "#7f7f7f", "fixed": "#ff7f0e",
              "rolling": "#2ca02c", "gate": "#d62728"}
    labels = {"none": "纯网格", "fixed": "固定阈值0.5",
              "rolling": "滚动最优θ", "gate": "Bandit门控"}

    n = len(seg_curves)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4.2 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for i, (curves, info) in enumerate(zip(seg_curves, seg_info)):
        ax = axes[i]
        ref = None
        for mode in MODES:
            df = curves.get(mode)
            if df is None or df.empty:
                continue
            ref = df
            ret = (df["total"] / df["total"].iloc[0] - 1) * 100
            ax.plot(df["date"], ret, color=colors[mode], lw=1.5, label=labels[mode])
        if ref is not None:
            bh = (ref["close"] / ref["close"].iloc[0] - 1) * 100
            ax.plot(ref["date"], bh, color="#bbbbbb", lw=1.0, ls=":", label="买入持有")
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_title(f"S{i + 1}: {info['range']} | {info['regime']}（持有 {info['bh_pct']:+.1f}%）")
        ax.set_ylabel("收益（%）")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"{code} 全周期滚动 walk-forward：四模式门控对比（每段严格样本外）", y=1.0)
    plt.tight_layout()
    out = DATA_DIR / "meta" / f"equity_walk_{code}{out_suffix}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("walk-forward 对比图已保存到 %s", out)


def run_rl_walk(
    codes: List[str],
    n_segments: int = 4,
    capital: Optional[float] = None,
    model: str = "lstm",
    epochs: int = 10,
    grid_n: int = 10,
    range_pct: float = 0.20,
    include_market: bool = True,
    include_dqn: bool = False,
    dqn_subset: int = 60,
    dqn_epochs: int = 5,
    include_hybrid: bool = False,
    trend_th: float = 0.03,
) -> pd.DataFrame:
    """全周期滚动 walk-forward：切 n_segments 段测试期（覆盖牛熊震荡）。

    每段严格用段前数据训练 gate 与 LSTM；资金跨段连续
    （上段期末资产作为下段初始资金）。输出分段对比 + 全周期汇总。
    include_market 控制 gate 状态特征是否含市场环境因子（指数环境+日历）。
    include_dqn 启用 DQN（真 RL，差分奖赏）门控作为第五模式。
    include_hybrid 启用混合门控（regime 切换：震荡日用 DQN、趋势日用 logistic），
    依赖 DQN，会自动启用 include_dqn；trend_th 为趋势判定阈值（|idx_ret_20d|）。
    """
    from backtest.engine import CostConfig, _perf_stats, run_engine
    from backtest.strategy import GridStrategy
    from data import store
    from features.builder import build_dataset
    from rl_gate.gate import make_buy_gate, train_gate
    from rl_gate.opportunity import build_day_features, build_opportunities, get_features

    if include_hybrid:
        include_dqn = True

    capital = capital if capital is not None else cfg.bt_capital
    cost = CostConfig()
    base_codes = _load_base_codes()
    meta_dir = DATA_DIR / "meta"
    suffix = "" if include_market else "_no_market"
    feats_list = get_features(include_market)

    modes_all = list(MODES)
    dqn_codes = None
    if include_dqn:
        import random as _rnd

        _rnd.seed(42)
        dqn_codes = _rnd.sample(base_codes, min(dqn_subset, len(base_codes)))
        modes_all.append("dqn")
        if include_hybrid:
            modes_all.append("hybrid")
        logger.info("DQN 模式启用：%d 只股票子集联合训练门控（hybrid=%s）",
                    len(dqn_codes), include_hybrid)

    logger.info(
        "构造沪深 300 机会样本（一次性，特征 %d 维，市场环境=%s）...",
        len(feats_list), include_market,
    )
    opp_all = build_opportunities(base_codes, with_market=include_market)
    if opp_all.empty:
        return pd.DataFrame()

    rows: List[Dict] = []
    for code in codes:
        bars = store.load_bars(code)
        if bars is None or bars.empty:
            continue
        sample = build_dataset(bars, window=cfg.window, horizon=cfg.horizon)
        if sample.empty:
            continue

        all_start, all_end = sample["date"].min(), sample["date"].max()
        # 预留前 20% 为初始训练期，其余等分为 n_segments 段测试期
        first_test = all_start + (all_end - all_start) * 0.2
        edges = [first_test + (all_end - first_test) * i / n_segments
                 for i in range(n_segments + 1)]
        logger.info(
            "=== %s | walk-forward %d 段 | 测试期 %s ~ %s ===",
            code, n_segments, first_test.date(), all_end.date(),
        )

        # 每模式跨段连续资金
        cur_capital = {m: capital for m in modes_all}
        agg = {m: {"init": capital, "final": capital} for m in modes_all}
        seg_curves, seg_info = [], []

        for i in range(n_segments):
            seg_start, seg_end = edges[i], edges[i + 1]
            bars_seg = bars[
                (bars["date"] >= seg_start) & (bars["date"] <= seg_end)
            ].reset_index(drop=True)
            if len(bars_seg) < 30:
                continue

            bh = float(bars_seg["close"].iloc[-1] / bars_seg["open"].iloc[0] - 1)
            regime = "牛市" if bh > 0.10 else ("熊市" if bh < -0.10 else "震荡")
            logger.info(
                "--- S%d: %s ~ %s | %s（持有 %+.1f%%）---",
                i + 1, seg_start.date(), seg_end.date(), regime, bh * 100,
            )

            # gate 训练（段前数据）
            gate_bundle = train_gate(opp_all[opp_all["date"] < seg_start], feats_list)

            # LSTM 段前训练 → 段信号
            ckpt_dir = DATA_DIR / "models" / "checkpoints" / f"walk_{code}_{model}_s{i + 1}"
            signals = _train_and_predict_segment(
                sample, seg_start, seg_end, model, ckpt_dir, epochs
            )

            # rolling θ：用段前最近 1 年搜索（训练数据不足 → 回退 0.5）
            pre_start = seg_start - pd.DateOffset(years=1)
            bars_pre = bars[
                (bars["date"] >= pre_start) & (bars["date"] < seg_start)
            ].reset_index(drop=True)
            sig_pre = signals.iloc[:0]
            if len(bars_pre) > 60 and len(sample[sample["date"] < pre_start]) > 100:
                try:
                    sig_pre = _train_and_predict_segment(
                        sample, pre_start, seg_start - pd.Timedelta(days=1),
                        model, ckpt_dir, epochs,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("滚动 θ 前置段训练失败（%s），θ 回退 0.5", exc)
            best_theta = (
                _search_theta(bars_pre, sig_pre, capital, cost, grid_n, range_pct)
                if len(sig_pre) > 0 else 0.5
            )

            # 四模式回测（资金跨段连续）
            curves = {}
            for mode, strat, sig in [
                ("none", GridStrategy(grid_n=grid_n, range_pct=range_pct, gate_on=False), None),
                ("fixed", GridStrategy(grid_n=grid_n, range_pct=range_pct,
                                       gate_on=True, gate_threshold=0.5), signals),
                ("rolling", GridStrategy(grid_n=grid_n, range_pct=range_pct,
                                         gate_on=True, gate_threshold=best_theta), signals),
            ]:
                res = run_engine(bars_seg, sig, strat,
                                 capital=cur_capital[mode], cost=cost)
                cur_capital[mode] = res.stats["final_value"]
                agg[mode]["final"] = cur_capital[mode]
                curves[mode] = res.equity_curve
                rows.append({
                    "code": code, "segment": f"S{i + 1} {regime}", "mode": mode,
                    "range": f"{seg_start.date()}~{seg_end.date()}",
                    "regime": regime, "gate_auc": round(gate_bundle["auc"], 4),
                    "best_theta": best_theta, **res.stats,
                })

            # gate 模式
            feats = build_day_features(bars, with_market=include_market)
            feats_seg = feats[
                (feats["date"] >= seg_start) & (feats["date"] <= seg_end)
            ]
            buy_gate = make_buy_gate(gate_bundle, feats_seg)
            res = run_engine(
                bars_seg, None,
                GridStrategy(grid_n=grid_n, range_pct=range_pct,
                             gate_on=False, buy_gate=buy_gate),
                capital=cur_capital["gate"], cost=cost,
            )
            cur_capital["gate"] = res.stats["final_value"]
            agg["gate"]["final"] = cur_capital["gate"]
            curves["gate"] = res.equity_curve
            rows.append({
                "code": code, "segment": f"S{i + 1} {regime}", "mode": "gate",
                "range": f"{seg_start.date()}~{seg_end.date()}",
                "regime": regime, "gate_auc": round(gate_bundle["auc"], 4),
                "best_theta": best_theta, **res.stats,
            })

            seg_curves.append(curves)
            seg_info.append({
                "range": f"{seg_start.date()} ~ {seg_end.date()}",
                "regime": regime, "bh_pct": bh * 100,
            })

            # DQN 模式（真 RL 门控：段前多股联合训练 → 测试段评估）
            if include_dqn and dqn_codes:
                from rl_gate.dqn import run_episode, train_dqn
                from rl_gate.env import GridBacktestEnv

                envs_pre = _build_envs(
                    dqn_codes, seg_start, grid_n, range_pct,
                    capital, cost, include_market,
                )
                if envs_pre:
                    dqn_agent = train_dqn(envs_pre, epochs=dqn_epochs)
                    feats_eval = build_day_features(
                        bars_seg, with_market=include_market
                    )
                    eval_env = GridBacktestEnv(
                        bars_seg, feats_eval, grid_n=grid_n, range_pct=range_pct,
                        capital=cur_capital["dqn"], cost=cost,
                    )
                    curve, eval_env = run_episode(eval_env, dqn_agent)
                    total_arr = curve["nav"].to_numpy(dtype=np.float64)
                    bh_dqn = float(curve["close"].iloc[-1] / curve["close"].iloc[0] - 1)
                    stats = _perf_stats(
                        eval_env.account, total_arr,
                        cur_capital["dqn"], max(len(curve) - 1, 1), bh_dqn,
                    )
                    cur_capital["dqn"] = stats["final_value"]
                    agg["dqn"]["final"] = cur_capital["dqn"]
                    seg_curves[-1]["dqn"] = curve.rename(columns={"nav": "total"})[
                        ["date", "total", "shares", "close"]
                    ]
                    rows.append({
                        "code": code, "segment": f"S{i + 1} {regime}", "mode": "dqn",
                        "range": f"{seg_start.date()}~{seg_end.date()}",
                        "regime": regime, "gate_auc": None, "best_theta": None,
                        **stats,
                    })

                    # 混合模式（regime 切换：趋势日 logistic、震荡日 DQN）
                    if include_hybrid:
                        regime_feats = build_day_features(
                            bars_seg, with_market=include_market
                        )
                        is_trend = regime_feats["idx_ret_20d"].abs() > trend_th
                        regime_by_date = dict(zip(regime_feats["date"], is_trend))
                        logistic_bg = make_buy_gate(gate_bundle, regime_feats)
                        env_h = GridBacktestEnv(
                            bars_seg, regime_feats, grid_n=grid_n, range_pct=range_pct,
                            capital=cur_capital["hybrid"], cost=cost,
                        )
                        curve_h, env_h = _run_hybrid_episode(
                            env_h, dqn_agent, logistic_bg, regime_by_date, trend_th
                        )
                        total_arr_h = curve_h["nav"].to_numpy(dtype=np.float64)
                        bh_h = float(curve_h["close"].iloc[-1] / curve_h["close"].iloc[0] - 1)
                        stats_h = _perf_stats(
                            env_h.account, total_arr_h,
                            cur_capital["hybrid"], max(len(curve_h) - 1, 1), bh_h,
                        )
                        cur_capital["hybrid"] = stats_h["final_value"]
                        agg["hybrid"]["final"] = cur_capital["hybrid"]
                        seg_curves[-1]["hybrid"] = curve_h.rename(columns={"nav": "total"})[
                            ["date", "total", "shares", "close"]
                        ]
                        n_trend_days = int(sum(regime_by_date.values()))
                        rows.append({
                            "code": code, "segment": f"S{i + 1} {regime}", "mode": "hybrid",
                            "range": f"{seg_start.date()}~{seg_end.date()}",
                            "regime": regime, "gate_auc": None, "best_theta": trend_th,
                            "n_trend_days": n_trend_days, **stats_h,
                        })

        _plot_walk_comparison(code, seg_curves, seg_info, out_suffix=suffix)

        # 全周期汇总行
        years = max((all_end - first_test).days / 365.0, 0.1)
        for mode in modes_all:
            total_ret = agg[mode]["final"] / agg[mode]["init"] - 1
            rows.append({
                "code": code, "segment": "ALL", "mode": mode,
                "range": f"{first_test.date()}~{all_end.date()}",
                "regime": "混合", "gate_auc": None, "best_theta": None,
                "capital": capital, "final_value": round(agg[mode]["final"], 2),
                "total_return_pct": round(total_ret * 100, 2),
                "annual_return_pct": round(((1 + total_ret) ** (1 / years) - 1) * 100, 2),
            })

    report = pd.DataFrame(rows)
    if report.empty:
        return report
    out = meta_dir / f"rl_walk_report{suffix}.csv"
    report.to_csv(out, index=False)
    logger.info("walk-forward 报告已保存到 %s（%d 行）", out, len(report))
    return report
