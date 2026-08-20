"""信号推送：企业微信 webhook 与 pushplus（微信公众号）双渠道。

凭证为敏感信息，约定存放在 cache/.env（已被 .gitignore 忽略）：
    WECOM_WEBHOOK=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx
    PUSHPLUS_TOKEN=xxxxxxxx
push_signal 按 cfg.push_channel 路由：auto 时优先 pushplus（有 token），
其次企业微信（有 webhook）。凭证未配置时仅打印不发送；发送异常只记日志不阻塞。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from config import DATA_DIR, cfg

logger = logging.getLogger(__name__)

ENV_FILE = DATA_DIR / ".env"


def load_env() -> None:
    """加载 cache/.env 中的 KEY=VALUE 到 os.environ（不覆盖已有环境变量）。"""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _webhook() -> str:
    load_env()
    return os.environ.get("WECOM_WEBHOOK", "") or cfg.wecom_webhook


def format_signal_message(signal: dict) -> str:
    """将信号 dict 格式化为通俗易懂的 markdown 推送消息。"""
    bullish = "偏多" in signal.get("action", "")
    color = "info" if bullish else "warning"
    chg = signal["predicted_change_pct"]
    direction = "涨" if chg >= 0 else "跌"
    prob_pct = round(signal["prob_close_up"] * 100)
    advice = "可以买入" if bullish else "只卖不买，先别加仓"
    return (
        f"**{signal['code']} 盘中信号**\n"
        f"> 今天 {signal['date']}，9:50 最新价 {signal['price_at_1000']} 元\n"
        f"> 模型判断：未来 30 分钟大概率{direction}，"
        f"到 {signal['predicted_close']} 元左右（{chg:+.2f}%）\n"
        f"> 上涨把握：{prob_pct}%\n"
        f"> 建议：<font color=\"{color}\">{advice}</font>"
    )


def push_wecom(text: str, webhook: str = "") -> bool:
    """发送 markdown 文本到企业微信机器人；未配置 webhook 时打印并返回 False。"""
    webhook = webhook or _webhook()
    if not webhook:
        logger.warning("WECOM_WEBHOOK 未配置，仅打印消息不推送：\n%s", text)
        return False
    try:
        import requests

        resp = requests.post(
            webhook,
            json={"msgtype": "markdown", "markdown": {"content": text}},
            timeout=10,
        )
        data = resp.json()
        if data.get("errcode") == 0:
            logger.info("企业微信推送成功")
            return True
        logger.error("企业微信推送失败: %s", data)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("企业微信推送异常: %s", exc)
        return False


def push_signal(signal: dict) -> bool:
    """按配置渠道推送一条信号（auto：pushplus 优先，其次企业微信）。"""
    text = format_signal_message(signal)
    title = f"盘中信号 {signal['code']} {signal['date']}"
    channel = cfg.push_channel
    if channel == "auto":
        if _pushplus_token():
            return push_pushplus(title, text)
        if _webhook():
            return push_wecom(text)
        logger.warning("未配置推送渠道（PUSHPLUS_TOKEN/WECOM_WEBHOOK），仅打印：\n%s", text)
        return False
    if channel == "pushplus":
        return push_pushplus(title, text)
    if channel == "wecom":
        return push_wecom(text)
    logger.warning("未知推送渠道 %s，仅打印：\n%s", channel, text)
    return False


# ---- pushplus（推送到关注的微信公众号，即普通微信）----


def _pushplus_token() -> str:
    load_env()
    return os.environ.get("PUSHPLUS_TOKEN", "") or cfg.pushplus_token


def _pushplus_topic() -> str:
    """一对多频道代码（可选）：配置后消息群发给频道内全部成员。"""
    load_env()
    return os.environ.get("PUSHPLUS_TOPIC", "") or cfg.pushplus_topic


def push_pushplus(
    title: str,
    text: str,
    token: str = "",
    topic: Optional[str] = None,
) -> bool:
    """发送 markdown 文本到 pushplus；未配置 token 时打印并返回 False。

    topic 为群组编码时走一对多群发（频道内全部成员经公众号接收）；
    注意不能传 channel=cp（那是企业微信应用渠道，需单独配置）。
    topic 缺省时读 PUSHPLUS_TOPIC 配置，仍未配置则发给个人。
    """
    token = token or _pushplus_token()
    if not token:
        logger.warning("PUSHPLUS_TOKEN 未配置，仅打印消息不推送：\n%s", text)
        return False
    topic = _pushplus_topic() if topic is None else topic
    payload = {
        "token": token,
        "title": title,
        "content": text,
        "template": "markdown",
    }
    if topic:
        payload["topic"] = topic
    try:
        import requests

        resp = requests.post(
            "http://www.pushplus.plus/send",
            json=payload,
            timeout=10,
        )
        data = resp.json()
        if data.get("code") == 200:
            logger.info("pushplus 推送成功")
            return True
        logger.error("pushplus 推送失败: %s", data)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("pushplus 推送异常: %s", exc)
        return False
