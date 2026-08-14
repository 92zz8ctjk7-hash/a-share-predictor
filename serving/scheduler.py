"""定时调度：每日盘中任务编排 + macOS launchd 集成。

run_daily_job 为单次任务：逐股预测 → 信号持久化 → 企业微信推送；
非交易日（无当日实时分钟数据）时预测返回 None，自动跳过。

install_scheduler 生成 launchd LaunchAgent（工作日定时触发 serve.sh），
serve.sh 为 cache/ 下的 wrapper 脚本（已被 .gitignore 忽略）。
"""

from __future__ import annotations

import logging
import stat
import subprocess
from pathlib import Path
from typing import List, Optional

from config import DATA_DIR, ROOT, cfg

logger = logging.getLogger(__name__)

PLIST_LABEL = "com.ashare.intraday-predict"
TRAIN_LABEL = "com.ashare.incremental-update"
LOG_DIR = DATA_DIR / "logs"


def run_daily_job(
    codes: List[str],
    frequency: Optional[str] = None,
    dry_run: bool = False,
) -> List[dict]:
    """执行一次盘中服务任务，返回生成的信号列表。

    dry_run=True 时只持久化并打印消息，不推送企业微信。
    """
    from serving.predict import predict_intraday
    from serving.push import format_signal_message, push_signal
    from serving.signal_store import save_signal

    frequency = frequency or cfg.min_frequency
    results: List[dict] = []
    for code in codes:
        try:
            signal = predict_intraday(code, frequency=frequency, realtime=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s 预测失败: %s", code, exc)
            continue
        if signal is None:
            logger.info("%s 跳过（非交易日或数据不足）", code)
            continue

        save_signal(signal)
        if dry_run:
            print(format_signal_message(signal))
        elif cfg.push_enabled:
            push_signal(signal)
        results.append(signal)

    logger.info("serve 任务完成：%d/%d 只产生信号", len(results), len(codes))
    return results


# ---- launchd 集成 ----


def build_serve_sh(codes: List[str], frequency: str) -> str:
    """生成预测任务 wrapper 脚本内容：加载本地环境并执行 serve。"""
    codes_arg = ",".join(codes)
    return (
        "#!/bin/bash\n"
        f"cd {ROOT}\n"
        f"exec {ROOT}/.venv/bin/python main.py serve "
        f"--codes {codes_arg} --frequency {frequency}\n"
    )


def build_train_sh(codes: List[str], frequency: str) -> str:
    """生成训练任务 wrapper 脚本内容：开盘前执行增量更新（数据刷新+重训）。"""
    codes_arg = ",".join(codes)
    return (
        "#!/bin/bash\n"
        f"cd {ROOT}\n"
        f"exec {ROOT}/.venv/bin/python main.py incremental-update "
        f"--codes {codes_arg} --frequency {frequency}\n"
    )


def build_plist(label: str, script_path: Path, hour: int, minute: int) -> str:
    """生成 launchd plist：工作日 hour:minute 触发指定脚本。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Weekday 1-5 = 周一到周五
    intervals = "\n".join(
        f"""        <dict>
            <key>Weekday</key>
            <integer>{wd}</integer>
            <key>Hour</key>
            <integer>{hour}</integer>
            <key>Minute</key>
            <integer>{minute}</integer>
        </dict>"""
        for wd in range(1, 6)
    )
    log_name = label.split(".")[-1]
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>{script_path}</string>
    </array>
    <key>StartCalendarInterval</key>
    <array>
{intervals}
    </array>
    <key>StandardOutPath</key>
    <string>{LOG_DIR / f"{log_name}.out.log"}</string>
    <key>StandardErrorPath</key>
    <string>{LOG_DIR / f"{log_name}.err.log"}</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""


def install_scheduler(
    codes: List[str],
    frequency: Optional[str] = None,
    load: bool = False,
) -> List[Path]:
    """生成两个 LaunchAgent：训练任务（开盘前增量更新）+ 预测任务（盘中推送）。

    load=True 时立即 launchctl load。返回 plist 安装路径列表。
    """
    frequency = frequency or cfg.min_frequency
    agents_dir = Path.home() / "Library" / "LaunchAgents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        # (标签, wrapper 脚本内容, 调度时间)
        (TRAIN_LABEL, build_train_sh(codes, frequency), cfg.train_hour, cfg.train_minute),
        (PLIST_LABEL, build_serve_sh(codes, frequency), cfg.schedule_hour, cfg.schedule_minute),
    ]

    paths: List[Path] = []
    for label, script_content, hour, minute in tasks:
        # wrapper 脚本（cache/ 下，已被 gitignore）
        script_path = DATA_DIR / f"{label.split('.')[-1]}.sh"
        script_path.write_text(script_content, encoding="utf-8")
        script_path.chmod(
            script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )
        logger.info("已生成调度脚本 %s", script_path)

        plist_path = agents_dir / f"{label}.plist"
        plist_path.write_text(build_plist(label, script_path, hour, minute), encoding="utf-8")
        logger.info("已生成 LaunchAgent %s（工作日 %02d:%02d）", plist_path, hour, minute)

        if load:
            subprocess.run(
                ["launchctl", "unload", str(plist_path)],
                capture_output=True, check=False,
            )
            subprocess.run(["launchctl", "load", str(plist_path)], check=True)
        paths.append(plist_path)

    if load:
        logger.info(
            "launchctl 已加载：训练 %02d:%02d 增量更新 → 预测 %02d:%02d 推送信号",
            cfg.train_hour, cfg.train_minute, cfg.schedule_hour, cfg.schedule_minute,
        )
    else:
        logger.info("未加载。启用请运行: launchctl load <plist路径>（或重新执行 --load）")
    return paths
