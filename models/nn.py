"""PyTorch 神经网络模型：MLP 与 LSTM。

统一支持双任务输出：
- 回归头：预测未来 N 日涨跌幅（连续值）
- 分类头：预测涨跌方向（0/1，含 sigmoid 后的概率）

forward 返回 (y_reg, y_cls_logit)。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """多层感知机。输入形状 (batch, n_features)。"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        y_reg = self.reg_head(h)
        y_cls_logit = self.cls_head(h)
        return y_reg, y_cls_logit


class LSTM(nn.Module):
    """LSTM 序列模型。输入形状 (batch, seq_len, n_features)。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, input_dim)
        out, (h_n, _) = self.lstm(x)
        # 使用最后一层最后一个时间步的输出
        last = out[:, -1, :]
        y_reg = self.reg_head(last)
        y_cls_logit = self.cls_head(last)
        return y_reg, y_cls_logit


class IntradayLSTM(nn.Module):
    """分钟级预测模型：LSTM 编码开盘 30 分钟序列 + 静态特征融合。

    forward(x_seq, x_static) -> (y_reg, y_cls_logit)
        x_seq   : (batch, seq_len, seq_input_dim) 开盘 30 分钟 5 分钟线序列
        x_static: (batch, static_dim) 日线技术特征 + 压力位 + 基座预测（级联）
    """

    def __init__(
        self,
        seq_input_dim: int = 4,
        static_dim: int = 21,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + static_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor):
        out, _ = self.lstm(x_seq)
        last = out[:, -1, :]  # 最后一个时间步（10:00 截面）
        h = self.fc(torch.cat([last, x_static], dim=1))
        y_reg = self.reg_head(h)
        y_cls_logit = self.cls_head(h)
        return y_reg, y_cls_logit


MODEL_REGISTRY = {
    "mlp": MLP,
    "lstm": LSTM,
    "intraday_lstm": IntradayLSTM,
}


def build_nn(model_name: str, input_dim: int, **kwargs) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_name}，可选 {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](input_dim=input_dim, **kwargs)