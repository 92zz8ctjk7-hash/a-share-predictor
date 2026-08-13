"""scikit-learn 基线模型。

提供：
- 回归：LinearRegression / GradientBoostingRegressor（预测涨跌幅）
- 分类：LogisticRegression / GradientBoostingClassifier（预测涨跌方向）

统一接口：fit() / predict()，predict 返回 (y_reg_pred, y_cls_pred, y_cls_score)。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


class BaselineEnsemble:
    """回归 + 分类两个独立模型的组合。"""

    def __init__(self, reg_name: str = "gbr", cls_name: str = "gbc"):
        self.reg_name = reg_name
        self.cls_name = cls_name
        self.scaler = StandardScaler()

        self.reg = self._make_regressor(reg_name)
        self.clf = self._make_classifier(cls_name)

    @staticmethod
    def _make_regressor(name: str):
        if name == "lr":
            return LinearRegression()
        if name == "gbr":
            return GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                random_state=42,
            )
        raise ValueError(f"未知回归模型: {name}")

    @staticmethod
    def _make_classifier(name: str):
        if name == "logreg":
            return LogisticRegression(max_iter=1000, random_state=42)
        if name == "gbc":
            return GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                random_state=42,
            )
        raise ValueError(f"未知分类模型: {name}")

    def fit(self, X: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray) -> "BaselineEnsemble":
        X = np.asarray(X, dtype=np.float32)
        # 标准化只针对回归/分类都适用，用缩放后的特征训练
        X_scaled = self.scaler.fit_transform(X)

        self.reg.fit(X_scaled, y_reg)
        self.clf.fit(X_scaled, y_cls)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X)

        y_reg_pred = self.reg.predict(X_scaled)
        y_cls_pred = self.clf.predict(X_scaled)
        y_cls_score = None
        if hasattr(self.clf, "predict_proba"):
            try:
                y_cls_score = self.clf.predict_proba(X_scaled)[:, 1]
            except Exception:  # noqa: BLE001
                y_cls_score = None
        return y_reg_pred, y_cls_pred, y_cls_score


REGISTRY: Dict[str, type] = {
    "baseline": BaselineEnsemble,
}