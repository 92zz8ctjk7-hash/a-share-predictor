"""训练与评估逻辑。

统一 NN 与 sklearn 基线的接口：
- train_model(bundle, cfg)  -> 返回训练好的模型对象
- evaluate_model(model, bundle, cfg) -> 返回测试集评估指标

NN 模型采用多任务损失：MSE(回归) + BCEWithLogits(分类)。
"""

from __future__ import annotations

import logging
import random
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:  # 未安装 torch 时，仅支持 sklearn 基线
    torch = None
    nn = None
    DataLoader = None

from config import Config
from models.baselines import BaselineEnsemble
from utils.metrics import evaluate_all

logger = logging.getLogger(__name__)


def get_device(cfg: Config) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg.device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_fn(batch):
    xs, yr, yc = zip(*batch)
    x = torch.stack(xs)
    y_reg = torch.cat(yr)
    y_cls = torch.cat(yc)
    return x, y_reg, y_cls


def _collate_min(batch):
    """分钟级样本 collate：(seq, static) 双输入。"""
    seqs, stats, yr, yc = zip(*batch)
    return (
        torch.stack(seqs),
        torch.stack(stats),
        torch.cat(yr),
        torch.cat(yc),
    )


class NNTrainer:
    """PyTorch 模型训练器，封装训练循环与预测。"""

    def __init__(self, cfg: Config):
        if torch is None:
            raise RuntimeError(
                "未安装 torch，无法训练神经网络模型。"
                "请安装 torch（可用镜像：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch），"
                "或使用基线模型：python main.py train --model baseline"
            )
        self.cfg = cfg
        self.device = get_device(cfg)
        set_seed(cfg.seed)

    def train(self, bundle, ckpt_dir: Optional[str] = None, save_every: int = 10) -> nn.Module:
        """训练网络。

        参数：
            bundle    : DataBundle（train/valid 为 Dataset）
            ckpt_dir  : checkpoint 保存目录；None 时不落盘。
                        val loss 最优存 best.pt，每 save_every 个 epoch 存 epoch_N.pt
            save_every: 定期 checkpoint 间隔（epoch）
        """
        from models.nn import MLP, LSTM  # 懒加载，避免未装 torch 时导入失败

        cfg = self.cfg
        input_dim = len(bundle.feature_names)

        if cfg.model == "lstm":
            net = LSTM(
                input_dim=input_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
            )
            seq_mode = True
        elif cfg.model == "mlp":
            net = MLP(input_dim=input_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
            seq_mode = False
        else:
            raise ValueError(f"未知 NN 模型: {cfg.model}")

        net = net.to(self.device)

        train_mode = "seq" if seq_mode else "flat"
        if bundle.train.mode != train_mode:
            raise ValueError(
                f"模型 {cfg.model} 需要 dataset mode='{train_mode}'，"
                f"但当前 bundle mode='{bundle.train.mode}'。请使用一致的 mode 构建数据。"
            )

        bundle_train = bundle.train
        bundle_valid = bundle.valid

        train_loader = DataLoader(
            bundle_train, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=_collate_fn,
        )
        valid_loader = DataLoader(
            bundle_valid, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=_collate_fn,
        )

        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        reg_criterion = nn.MSELoss()
        cls_criterion = nn.BCEWithLogitsLoss()

        best_valid_loss = float("inf")
        best_state = None
        ckpt_path = None
        if ckpt_dir is not None:
            from pathlib import Path

            ckpt_path = Path(ckpt_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)

        def _save_ckpt(path, state) -> None:
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in state.items()},
                    "model": cfg.model,
                    "input_dim": input_dim,
                    "feature_names": list(bundle.feature_names),
                    "hidden_dim": cfg.hidden_dim,
                    "num_layers": cfg.num_layers,
                    "dropout": cfg.dropout,
                },
                path,
            )

        for epoch in range(1, cfg.epochs + 1):
            net.train()
            total_loss = 0.0
            n_batches = 0
            for x, y_reg, y_cls in train_loader:
                x = x.to(self.device)
                y_reg = y_reg.to(self.device).view(-1, 1)
                y_cls = y_cls.to(self.device).float().view(-1, 1)

                optimizer.zero_grad()
                pred_reg, pred_cls_logit = net(x)
                loss_reg = reg_criterion(pred_reg, y_reg)
                loss_cls = cls_criterion(pred_cls_logit, y_cls)
                loss = loss_reg + loss_cls
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # 验证
            net.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for x, y_reg, y_cls in valid_loader:
                    x = x.to(self.device)
                    y_reg = y_reg.to(self.device).view(-1, 1)
                    y_cls = y_cls.to(self.device).float().view(-1, 1)
                    pred_reg, pred_cls_logit = net(x)
                    loss = reg_criterion(pred_reg, y_reg) + cls_criterion(pred_cls_logit, y_cls)
                    val_loss += float(loss.item())
                    val_batches += 1

            val_loss = val_loss / max(val_batches, 1)
            logger.info(
                "Epoch %d/%d | train_loss=%.4f valid_loss=%.4f",
                epoch, cfg.epochs, avg_loss, val_loss,
            )

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                if ckpt_path is not None:
                    _save_ckpt(ckpt_path / "best.pt", best_state)

            if ckpt_path is not None and epoch % max(save_every, 1) == 0:
                _save_ckpt(
                    ckpt_path / f"epoch_{epoch}.pt",
                    net.state_dict(),
                )

        if best_state is not None:
            net.load_state_dict(best_state)

        self.model = net
        return net

    def predict(self, bundle, split: str = "test"):
        net = self.model
        net = net.to(self.device)
        net.eval()

        dataset = getattr(bundle, split)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=_collate_fn)

        reg_preds = []
        cls_preds = []
        cls_scores = []
        with torch.no_grad():
            for x, _, _ in loader:
                x = x.to(self.device)
                pred_reg, pred_cls_logit = net(x)
                y_reg = pred_reg.squeeze(1).cpu().numpy()
                score = torch.sigmoid(pred_cls_logit).squeeze(1).cpu().numpy()
                y_cls = (score >= 0.5).astype(np.int64)

                reg_preds.append(y_reg)
                cls_preds.append(y_cls)
                cls_scores.append(score)

        return (
            np.concatenate(reg_preds),
            np.concatenate(cls_preds),
            np.concatenate(cls_scores),
        )


def load_nn_checkpoint(path: str):
    """加载 NN checkpoint，返回 (net, meta)。

    meta 为保存时的模型结构信息（model/input_dim/feature_names 等），
    可直接用于重建网络与对齐特征列。
    """
    from models.nn import build_nn

    ckpt = torch.load(path, map_location="cpu")
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    # MLP 无 num_layers 参数，按模型类型裁剪构造参数
    kwargs = {
        "hidden_dim": meta["hidden_dim"],
        "dropout": meta.get("dropout", 0.2),
    }
    if meta["model"] == "lstm":
        kwargs["num_layers"] = meta.get("num_layers", 2)
    net = build_nn(meta["model"], input_dim=meta["input_dim"], **kwargs)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, meta


def nn_predict(
    net,
    X: np.ndarray,
    mode: str = "flat",
    seq_len: int = 10,
    batch_size: int = 1024,
    device=None,
):
    """用 NN 模型对特征矩阵推理，返回 (pred_reg, prob_up)。

    mode="seq" 时 X 视为时序特征矩阵，内部滑窗构造序列后逐样本预测。
    """
    if device is None:
        device = torch.device("cpu")
    net = net.to(device)
    net.eval()

    if mode == "seq":
        total = len(X) - seq_len + 1
        if total <= 0:
            return np.array([]), np.array([])
        batches = [
            np.stack([X[i:i + seq_len] for i in range(s, min(s + batch_size, total))])
            for s in range(0, total, batch_size)
        ]
    else:
        batches = [X[s:s + batch_size] for s in range(0, len(X), batch_size)]

    preds, probs = [], []
    with torch.no_grad():
        for xb in batches:
            xs = torch.tensor(xb, dtype=torch.float32).to(device)
            pred_reg, pred_cls_logit = net(xs)
            preds.append(pred_reg.squeeze(1).cpu().numpy())
            probs.append(torch.sigmoid(pred_cls_logit).squeeze(1).cpu().numpy())

    return np.concatenate(preds), np.concatenate(probs)


def predict_base_all(bundle, cfg: Config, model_path: str) -> "pd.DataFrame":
    """用训练好的日线基座模型对全部样本推理，输出 (date, code, base_pred)。

    返回的 date 为「预测目标日」：样本日期按 code 分组后 shift(-1)，
    即基座对 horizon=1 目标交易日的涨跌预测，可直接与分钟样本按 date 对齐。
    """
    import pandas as pd

    from models.nn import build_nn

    device = get_device(cfg)
    input_dim = len(bundle.feature_names)

    if cfg.model == "lstm":
        net = build_nn(
            "lstm", input_dim=input_dim,
            hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    elif cfg.model == "mlp":
        net = build_nn(
            "mlp", input_dim=input_dim,
            hidden_dim=cfg.hidden_dim, dropout=cfg.dropout,
        )
    else:
        raise ValueError(f"基座模型仅支持 lstm/mlp: {cfg.model}")

    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net = net.to(device)
    net.eval()

    frames = []
    for split in ("train", "valid", "test"):
        ds = getattr(bundle, split)
        df = getattr(bundle, f"{split}_df")
        loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=_collate_fn)

        preds = []
        with torch.no_grad():
            for x, _, _ in loader:
                x = x.to(device)
                pred, _ = net(x)
                preds.append(pred.squeeze(1).cpu().numpy())
        if not preds:
            continue

        sub = df[["date", "code"]].copy()
        sub["base_pred"] = np.concatenate(preds)
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["date", "code", "base_pred"])

    out = pd.concat(frames, ignore_index=True)
    # 样本日期 T-1 的预测对应目标交易日 T
    out = out.sort_values(["code", "date"]).reset_index(drop=True)
    out["date"] = out.groupby("code")["date"].shift(-1)
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    logger.info("基座推理完成：共 %d 条预测（已对齐到目标交易日）", len(out))
    return out[["date", "code", "base_pred"]]


class MinNNTrainer:
    """分钟级模型训练器（IntradayLSTM：序列 + 静态特征）。"""

    def __init__(self, cfg: Config):
        if torch is None:
            raise RuntimeError("未安装 torch，无法训练神经网络模型")
        self.cfg = cfg
        self.device = get_device(cfg)
        set_seed(cfg.seed)

    def train(self, bundle) -> nn.Module:
        from models.nn import IntradayLSTM

        cfg = self.cfg
        net = IntradayLSTM(
            seq_input_dim=bundle.seq_input_dim,
            static_dim=len(bundle.static_cols),
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(self.device)

        train_loader = DataLoader(
            bundle.train, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=_collate_min,
        )
        valid_loader = DataLoader(
            bundle.valid, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=_collate_min,
        )

        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        reg_criterion = nn.MSELoss()
        cls_criterion = nn.BCEWithLogitsLoss()

        best_valid_loss = float("inf")
        best_state = None

        for epoch in range(1, cfg.epochs + 1):
            net.train()
            total_loss = 0.0
            n_batches = 0
            for x_seq, x_static, y_reg, y_cls in train_loader:
                x_seq = x_seq.to(self.device)
                x_static = x_static.to(self.device)
                y_reg = y_reg.to(self.device).view(-1, 1)
                y_cls = y_cls.to(self.device).float().view(-1, 1)

                optimizer.zero_grad()
                pred_reg, pred_cls_logit = net(x_seq, x_static)
                loss = reg_criterion(pred_reg, y_reg) + cls_criterion(pred_cls_logit, y_cls)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            net.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for x_seq, x_static, y_reg, y_cls in valid_loader:
                    x_seq = x_seq.to(self.device)
                    x_static = x_static.to(self.device)
                    y_reg = y_reg.to(self.device).view(-1, 1)
                    y_cls = y_cls.to(self.device).float().view(-1, 1)
                    pred_reg, pred_cls_logit = net(x_seq, x_static)
                    loss = reg_criterion(pred_reg, y_reg) + cls_criterion(pred_cls_logit, y_cls)
                    val_loss += float(loss.item())
                    val_batches += 1

            val_loss = val_loss / max(val_batches, 1)
            logger.info(
                "Epoch %d/%d | train_loss=%.4f valid_loss=%.4f",
                epoch, cfg.epochs, total_loss / max(n_batches, 1), val_loss,
            )

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if best_state is not None:
            net.load_state_dict(best_state)

        self.model = net
        return net

    def predict(self, bundle, split: str = "test"):
        """返回 (y_reg_pred, y_cls_pred, y_cls_score)。"""
        net = self.model.to(self.device)
        net.eval()

        dataset = getattr(bundle, split)
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=False,
            collate_fn=_collate_min,
        )

        reg_preds = []
        cls_preds = []
        cls_scores = []
        with torch.no_grad():
            for x_seq, x_static, _, _ in loader:
                pred_reg, pred_cls_logit = net(
                    x_seq.to(self.device), x_static.to(self.device)
                )
                reg_preds.append(pred_reg.squeeze(1).cpu().numpy())
                score = torch.sigmoid(pred_cls_logit).squeeze(1).cpu().numpy()
                cls_scores.append(score)
                cls_preds.append((score >= 0.5).astype(np.int64))

        return (
            np.concatenate(reg_preds),
            np.concatenate(cls_preds),
            np.concatenate(cls_scores),
        )


def train_min_model(bundle, cfg: Config):
    """训练分钟级模型，返回 (trainer, metrics)。"""
    trainer = MinNNTrainer(cfg)
    trainer.train(bundle)

    y_reg_pred, y_cls_pred, y_cls_score = trainer.predict(bundle, "test")
    y_reg_true = bundle.test_df["label_rest"].to_numpy(dtype=np.float32)
    y_cls_true = bundle.test_df["label_cls_rest"].to_numpy(dtype=np.int64)
    metrics = evaluate_all(y_reg_true, y_reg_pred, y_cls_true, y_cls_pred, y_cls_score)
    return trainer, metrics


def train_model(
    bundle,
    cfg: Config,
    ckpt_dir: Optional[str] = None,
    save_every: int = 10,
):
    """训练入口，根据 cfg.model 选择 NN 或基线。

    ckpt_dir 非 None 时：NN 保存 best.pt / epoch_N.pt 到该目录；
    基线模型（无 epoch 概念）在训练后保存 best.joblib。
    """
    if cfg.model in ("mlp", "lstm"):
        trainer = NNTrainer(cfg)
        model = trainer.train(bundle, ckpt_dir=ckpt_dir, save_every=save_every)
        return ("nn", trainer)
    if cfg.model == "baseline":
        X, y_reg, y_cls = bundle.X_y("train")
        model = BaselineEnsemble()
        model.fit(X, y_reg, y_cls)
        if ckpt_dir is not None:
            import joblib
            from pathlib import Path

            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            joblib.dump(model, Path(ckpt_dir) / "best.joblib")
        return ("baseline", model)

    raise ValueError(f"未知模型类型: {cfg.model}")


def evaluate_model(model_info, bundle, cfg: Config) -> Dict[str, Dict[str, float]]:
    """评估模型在测试集上的表现。"""
    kind, model = model_info

    y_reg_true = bundle.test_df["label_reg"].to_numpy(dtype=np.float32)
    y_cls_true = bundle.test_df["label_cls"].to_numpy(dtype=np.int64)

    if kind == "nn":
        y_reg_pred, y_cls_pred, y_cls_score = model.predict(bundle, "test")
    elif kind == "baseline":
        X, _, _ = bundle.X_y("test")
        y_reg_pred, y_cls_pred, y_cls_score = model.predict(X)
    else:
        raise ValueError(f"未知模型类型: {kind}")

    return evaluate_all(y_reg_true, y_reg_pred, y_cls_true, y_cls_pred, y_cls_score)


def save_model(model_info, path: str) -> None:
    """保存模型到磁盘。NN 保存 state_dict，基线用 joblib。"""
    kind, model = model_info
    if kind == "nn":
        torch.save(model.model.state_dict(), path)
    else:
        import joblib
        joblib.dump(model, path)
    logger.info("模型已保存到 %s", path)


def save_min_model(trainer, bundle, path: str) -> None:
    """保存分钟级模型：state_dict + 静态特征列元信息（推理时对齐用）。"""
    torch.save(
        {
            "state_dict": trainer.model.state_dict(),
            "static_cols": bundle.static_cols,
            "seq_input_dim": bundle.seq_input_dim,
        },
        path,
    )
    logger.info("分钟模型已保存到 %s（static_cols=%d 个）", path, len(bundle.static_cols))