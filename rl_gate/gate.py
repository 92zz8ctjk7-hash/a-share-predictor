"""门控策略训练与推理：逻辑回归学习「何时放行网格买入」。

训练样本 = 沪深 300 买入机会（市场状态特征 + 未来 5 日收益标签），
时间序列切分（前 80% 训练、后 20% 验证，防泄漏），输出验证 AUC。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)

GATE_PATH = DATA_DIR / "models" / "gate_logistic.joblib"


def train_gate(samples: pd.DataFrame, features) -> Dict:
    """训练门控模型，返回 {'model', 'auc', 'coef'}；保存到 GATE_PATH。

    时间序列切分：按日期排序后前 80% 训练、后 20% 验证。
    """
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    s = samples.sort_values("date").reset_index(drop=True)
    cut = int(len(s) * 0.8)
    tr, va = s.iloc[:cut], s.iloc[cut:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(tr[features].to_numpy(dtype=float))
    y_tr = tr["y"].to_numpy()
    X_va = scaler.transform(va[features].to_numpy(dtype=float))
    y_va = va["y"].to_numpy()

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)

    auc = float(roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1]))
    logger.info(
        "gate 训练完成：训练 %d / 验证 %d 条，验证 AUC=%.4f",
        len(tr), len(va), auc,
    )
    coef = pd.Series(clf.coef_[0], index=list(features))
    logger.info("gate 系数（正=利于放行）：\n%s", coef.round(3).to_string())

    joblib.dump(
        {"model": clf, "scaler": scaler, "features": list(features)}, GATE_PATH
    )
    return {"model": clf, "scaler": scaler, "features": list(features), "auc": auc, "coef": coef}


def load_gate() -> Dict:
    """加载已训练的门控模型。"""
    import joblib

    if not GATE_PATH.exists():
        raise FileNotFoundError(f"门控模型不存在: {GATE_PATH}，请先运行 rl-backtest")
    return joblib.load(GATE_PATH)


def gate_allows(bundle: Dict, features_df: pd.DataFrame) -> pd.Series:
    """批量判断各日期是否放行买入，返回与 features_df 对齐的 bool Series。"""
    X = bundle["scaler"].transform(features_df[bundle["features"]].to_numpy(dtype=float))
    proba = bundle["model"].predict_proba(X)[:, 1]
    return pd.Series(proba >= 0.5, index=features_df.index)


def make_buy_gate(bundle: Dict, features_df: pd.DataFrame):
    """构造引擎可用的 buy_gate 回调：day -> bool（未知日期默认放行）。"""
    allows = gate_allows(bundle, features_df)
    by_date = dict(zip(features_df["date"], allows))

    def buy_gate(day) -> bool:
        return bool(by_date.get(day, True))

    return buy_gate
