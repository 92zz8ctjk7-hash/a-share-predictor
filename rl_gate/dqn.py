"""DQN 门控 agent：差分奖赏 + 经验回放 + 目标网络。

在 GridBacktestEnv 上训练「何时放行网格买入」的门控策略：
- 动作二值：{0: 拦截, 1: 放行}
- 奖赏：差分超额收益 - 换手惩罚（见 env.py）
- 多股票环境联合训练（跨股共享门控规律，解决单股样本不足）
"""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """经验回放缓冲区。"""

    def __init__(self, capacity: int = 200000):
        self.buf: deque = deque(maxlen=capacity)

    def add(self, s, a, r, s_next, done) -> None:
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


class QNet(nn.Module):
    """小型 Q 网络（低参数防小样本过拟合），支持 n_actions 动作。"""

    def __init__(self, state_dim: int, n_actions: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN：ε-greedy 采样 + 目标网络软更新。"""

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps_start: float = 0.5,
        eps_min: float = 0.05,
        eps_decay: float = 0.995,
    ):
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.n_actions = n_actions
        self.q = QNet(state_dim, n_actions)
        self.target = QNet(state_dim, n_actions)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.step_count = 0

    def act(self, s: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            qv = self.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
            return int(qv.argmax(dim=1).item())

    def decay_eps(self) -> None:
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def train_step(self, batch) -> float:
        s, a, r, s2, done = batch
        qv = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            tgt = r + self.gamma * self.target(s2).max(dim=1)[0] * (1 - done)
        loss = nn.functional.mse_loss(qv, tgt)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()
        self.step_count += 1
        if self.step_count % 500 == 0:  # 目标网络定期硬更新
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())


def train_dqn(
    envs: Dict[str, object],
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
) -> DQNAgent:
    """在多股票环境上训练 DQN 门控（动作数由环境推断）。

    每个 epoch：全部环境各跑一个 episode（ε-greedy 采样入 buffer），
    然后按 buffer 规模训练若干 batch。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    codes = list(envs)
    probe = envs[codes[0]]
    state_dim = probe.state_dim
    n_actions = getattr(probe, "n_actions", 2)
    agent = DQNAgent(state_dim, n_actions=n_actions, lr=lr)
    buffer = ReplayBuffer()

    for ep in range(epochs):
        total_r = 0.0
        for code in codes:
            env = envs[code]
            s = env.reset()
            done = False
            while not done:
                a = agent.act(s, explore=True)
                s2, r, done, _ = env.step(a)
                buffer.add(s, a, r, s2, done)
                total_r += r
                s = s2
            agent.decay_eps()

        n_batches = max(len(buffer) // batch_size, 100)
        losses = [agent.train_step(buffer.sample(batch_size)) for _ in range(n_batches)]
        logger.info(
            "DQN epoch %d/%d：buffer=%d loss=%.4f ε=%.2f Σr=%.1f",
            ep + 1, epochs, len(buffer), float(np.mean(losses)), agent.eps, total_r,
        )
    return agent


def run_episode(env, agent: DQNAgent) -> Tuple[pd.DataFrame, object]:
    """贪心策略评估一个 episode，返回 (资金曲线 DataFrame, env)。"""
    s = env.reset()
    done = False
    rows: List[Dict] = []
    while not done:
        a = agent.act(s, explore=False)
        s, r, done, info = env.step(a)
        rows.append(info)
    curve = pd.DataFrame(rows)
    return curve, env
