"""P1 轨迹解码器：自回归高斯模型，拟合日内分钟轨迹。

核心思想：日内轨迹 = 每分钟条件分布的「积分」。
对涨跌序列 {r_t} 按自回归分解联合分布：
    P(r_1..r_T | C) = Π_t P(r_t | r_{<t}, C)
每一步输出高斯参数 (μ_t, σ_t)，拟合轨迹 = 条件期望路径 {μ_t}，
σ_t 刻画该时刻的不确定度（供 P2 RL 强度控制器使用）。

训练目标：高斯负对数似然（NLL）
    L = mean[ log σ_t + (r_t - μ_t)² / (2σ_t²) ]

推理：给定当日已发生的 r_{<t0}，滚动前向输出剩余时段期望路径。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 自回归回看长度（输入最近 L 根 bar 的涨跌）
LOOKBACK = 12


def build_daily_returns(min_bars: pd.DataFrame) -> pd.DataFrame:
    """从 5 分钟 bar 构造每日涨跌序列 r_t（相对前一根，首根相对前收）。

    返回长表：date, time, r（每根 bar 的涨跌比例，非百分比）。
    """
    out = []
    for d, g in min_bars.groupby("date"):
        g = g.sort_values("time").reset_index(drop=True)
        close = g["close"].to_numpy(dtype=np.float64)
        prev = np.concatenate([[np.nan], close[:-1]])
        # 首根相对前一日收盘（若可得）
        r = close / prev - 1.0
        r[0] = 0.0 if np.isnan(r[0]) else r[0]
        out.append(pd.DataFrame({"date": d, "time": g["time"], "r": r}))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def build_trajectory_samples(
    ret_df: pd.DataFrame,
    base_preds: Optional[pd.DataFrame] = None,
    code: Optional[str] = None,
    lookback: int = LOOKBACK,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """构造自回归样本：输入最近 lookback 根涨跌 + 条件 base_pred，目标 r_t。

    返回 (r_seq, cond, target, time_idx, meta)：
        r_seq   : (N, lookback)
        cond    : (N, 1)  base_pred（无则 0）
        target  : (N,)    r_t
        time_idx: (N,)    目标 bar 在当日的序号（0-47，时刻 embedding 用）
        meta    : date/time 对照表
    """
    bp = {}
    if base_preds is not None and code is not None:
        hit = base_preds[base_preds["code"] == code]
        bp = dict(zip(pd.to_datetime(hit["date"]).dt.date, hit["base_pred"]))

    seqs, conds, targets, tidxs, meta = [], [], [], [], []
    for d, g in ret_df.groupby("date"):
        r = g["r"].to_numpy(dtype=np.float32)
        if len(r) < lookback + 1:
            continue
        bpv = float(bp.get(d, 0.0))
        for t in range(lookback, len(r)):
            seqs.append(r[t - lookback:t])
            conds.append([bpv])
            targets.append(r[t])
            tidxs.append(t)
            meta.append({"date": d, "time": g["time"].iloc[t], "idx": t})
    if not seqs:
        return (np.empty((0, lookback)), np.empty((0, 1)), np.empty(0),
                np.empty(0, dtype=np.int64), pd.DataFrame())
    return (np.array(seqs, dtype=np.float32),
            np.array(conds, dtype=np.float32),
            np.array(targets, dtype=np.float32),
            np.array(tidxs, dtype=np.int64),
            pd.DataFrame(meta))


class TrajectoryDecoder(nn.Module):
    """自回归高斯轨迹解码器（含时刻 embedding）。

    输入最近 lookback 根涨跌序列 + 条件 + 目标时刻，输出下一 bar 的 (μ, log σ)。
    时刻 embedding 使模型能学习日内波动结构（开盘/收盘波动大等）。
    """

    def __init__(self, cond_dim: int = 1, lookback: int = LOOKBACK,
                 hidden: int = 64, n_times: int = 48, time_emb_dim: int = 8):
        super().__init__()
        self.lookback = lookback
        self.time_emb = nn.Embedding(n_times, time_emb_dim)
        self.lstm = nn.LSTM(1 + cond_dim, hidden, num_layers=1, batch_first=True)
        self.mu_head = nn.Linear(hidden + time_emb_dim, 1)
        self.log_sigma_head = nn.Linear(hidden + time_emb_dim, 1)

    def forward(self, r_seq: torch.Tensor, cond: torch.Tensor, time_idx: torch.Tensor):
        """r_seq:(B,L) cond:(B,cond_dim) time_idx:(B,) → (μ, log σ) 各 (B,)。"""
        B, L = r_seq.shape
        cond_exp = cond.unsqueeze(1).expand(B, L, -1)
        x = torch.cat([r_seq.unsqueeze(-1), cond_exp], dim=-1)
        h, _ = self.lstm(x)
        h_last = h[:, -1, :]
        te = self.time_emb(time_idx)  # (B, time_emb_dim)
        feat = torch.cat([h_last, te], dim=-1)
        mu = self.mu_head(feat).squeeze(-1)
        log_sigma = self.log_sigma_head(feat).squeeze(-1).clamp(-6.0, 2.0)
        return mu, log_sigma


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor):
    """高斯负对数似然。"""
    sigma2 = torch.exp(2 * log_sigma)
    return (log_sigma + (target - mu) ** 2 / (2 * sigma2)).mean()


def train_trajectory(
    r_seq: np.ndarray, cond: np.ndarray, target: np.ndarray,
    time_idx: np.ndarray,
    epochs: int = 50, batch_size: int = 512, lr: float = 1e-4,
    val_frac: float = 0.15, seed: int = 42,
) -> Tuple[TrajectoryDecoder, Dict]:
    """训练轨迹解码器（时间序列尾部切验证集），返回 (model, history)。"""
    torch.manual_seed(seed)
    n = len(r_seq)
    cut = int(n * (1 - val_frac))

    Xtr, Ctr, Ytr, Ttr = r_seq[:cut], cond[:cut], target[:cut], time_idx[:cut]
    Xva, Cva, Yva, Tva = r_seq[cut:], cond[cut:], target[cut:], time_idx[cut:]

    model = TrajectoryDecoder(cond_dim=cond.shape[1], lookback=r_seq.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xt = torch.tensor(Xtr); Ct = torch.tensor(Ctr); Yt = torch.tensor(Ytr)
    Tt = torch.tensor(Ttr, dtype=torch.long)
    Xv = torch.tensor(Xva); Cv = torch.tensor(Cva); Yv = torch.tensor(Yva)
    Tv = torch.tensor(Tva, dtype=torch.long)

    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0; nb = 0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            mu, ls = model(Xt[idx], Ct[idx], Tt[idx])
            loss = gaussian_nll(mu, ls, Yt[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            mu_v, ls_v = model(Xv, Cv, Tv)
            val_loss = gaussian_nll(mu_v, ls_v, Yv).item()
        history.append({"epoch": ep + 1, "train_nll": tot / nb, "val_nll": val_loss})
        if (ep + 1) % 10 == 0:
            logger.info("轨迹解码器 epoch %d/%d train_nll=%.5f val_nll=%.5f",
                        ep + 1, epochs, tot / nb, val_loss)
    return model, {"history": history}


def reconstruct(
    model: TrajectoryDecoder, r_full: np.ndarray, cond_val: float,
) -> pd.DataFrame:
    """teacher forcing 重建：给定全天 r，用真实历史预测每 bar 的 μ/σ。

    用于评估拟合效果（对比 μ 与实际 r）。返回 idx/mu/sigma/actual。
    """
    model.eval()
    L = model.lookback
    r_full = r_full.astype(np.float32)
    T = len(r_full)
    if T < L + 1:
        return pd.DataFrame()
    seqs = np.stack([r_full[t - L:t] for t in range(L, T)])
    conds = np.full((T - L, 1), cond_val, dtype=np.float32)
    tidx = np.arange(L, T, dtype=np.int64)
    with torch.no_grad():
        mu, ls = model(torch.tensor(seqs), torch.tensor(conds),
                       torch.tensor(tidx, dtype=torch.long))
    return pd.DataFrame({
        "idx": np.arange(L, T),
        "mu": mu.numpy(),
        "sigma": np.exp(ls.numpy()),
        "actual": r_full[L:],
    })


def predict_future(
    model: TrajectoryDecoder, r_observed: np.ndarray, cond_val: float,
    day_len: int = 48,
) -> pd.DataFrame:
    """给定已发生 r_observed，自回归滚动前向预测剩余轨迹。

    未来部分用预测均值 μ 作为下一步输入（期望路径）。返回 idx/mu/sigma。
    """
    model.eval()
    L = model.lookback
    r_ext = list(r_observed.astype(np.float32))
    rows = []
    with torch.no_grad():
        while len(r_ext) < day_len:
            t_idx = len(r_ext)
            seq = torch.tensor(np.array(r_ext[-L:], dtype=np.float32)).unsqueeze(0)
            cond_t = torch.tensor([[cond_val]], dtype=torch.float32)
            tidx_t = torch.tensor([t_idx], dtype=torch.long)
            mu, ls = model(seq, cond_t, tidx_t)
            rows.append({
                "idx": t_idx,
                "mu": float(mu.item()),
                "sigma": float(torch.exp(ls).item()),
            })
            r_ext.append(float(mu.item()))  # 自回归填充
    return pd.DataFrame(rows)


# ---- 30 分钟窗口版本（信号密度更低、信噪比更高）----

# 每个 30 分钟窗口 = 6 根 5 分钟 bar；一天 48 bar = 8 窗口
BARS_PER_WINDOW = 6


def build_window_returns(min_bars: pd.DataFrame, bars_per_window: int = BARS_PER_WINDOW) -> pd.DataFrame:
    """将每日 bar 切成 30 分钟窗口，计算每窗口涨跌 r_w。

    r_w = 窗口末 bar 收盘 / 前一窗口末 bar 收盘 - 1（首窗口相对前日收盘）。
    返回长表：date, win_idx, r。
    """
    out = []
    for d, g in min_bars.groupby("date"):
        g = g.sort_values("time").reset_index(drop=True)
        close = g["close"].to_numpy(dtype=np.float64)
        n_win = len(close) // bars_per_window
        if n_win < 2:
            continue
        # 每窗口末 bar 的收盘价
        win_close = np.array([close[(w + 1) * bars_per_window - 1] for w in range(n_win)])
        prev = np.concatenate([[np.nan], win_close[:-1]])
        r = win_close / prev - 1.0
        r[0] = 0.0 if np.isnan(r[0]) else r[0]
        out.append(pd.DataFrame({"date": d, "win_idx": np.arange(n_win), "r": r}))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def build_window_trajectory_samples(
    win_ret_df: pd.DataFrame,
    base_preds: Optional[pd.DataFrame] = None,
    code: Optional[str] = None,
    lookback: int = 3,
    n_windows: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """构造窗口级自回归样本：输入最近 lookback 个窗口涨跌 + 条件，目标 r_w。

    返回 (r_seq, cond, target, win_idx, meta)，同 bar 版本但窗口级。
    """
    bp = {}
    if base_preds is not None and code is not None:
        hit = base_preds[base_preds["code"] == code]
        bp = dict(zip(pd.to_datetime(hit["date"]).dt.date, hit["base_pred"]))

    seqs, conds, targets, widxs, meta = [], [], [], [], []
    for d, g in win_ret_df.groupby("date"):
        r = g.sort_values("win_idx")["r"].to_numpy(dtype=np.float32)
        if len(r) < lookback + 1:
            continue
        bpv = float(bp.get(d, 0.0))
        for w in range(lookback, len(r)):
            seqs.append(r[w - lookback:w])
            conds.append([bpv])
            targets.append(r[w])
            widxs.append(w)
            meta.append({"date": d, "win_idx": w})
    if not seqs:
        return (np.empty((0, lookback)), np.empty((0, 1)), np.empty(0),
                np.empty(0, dtype=np.int64), pd.DataFrame())
    return (np.array(seqs, dtype=np.float32),
            np.array(conds, dtype=np.float32),
            np.array(targets, dtype=np.float32),
            np.array(widxs, dtype=np.int64),
            pd.DataFrame(meta))
