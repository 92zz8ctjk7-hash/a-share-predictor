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
GUARD_LABEL = "com.ashare.serve-guard"
NEXTDAY_LABEL = "com.ashare.nextday-predict"
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


INTRADAY_LABEL = "com.ashare.intraday-signal"


def build_intraday_sh(codes: List[str]) -> str:
    """生成盘中信号任务 wrapper 脚本内容：交易日 10:01 推送盘中信号。"""
    codes_arg = ",".join(codes)
    return (
        "#!/bin/bash\n"
        f"cd {ROOT}\n"
        f"exec {ROOT}/.venv/bin/python main.py serve-intraday "
        f"--codes {codes_arg}\n"
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


def install_intraday_scheduler(codes: List[str], load: bool = False) -> Path:
    """安装盘中信号定时任务（交易日 10:01 推送拟合轨迹+强度信号）。

    load=True 时立即 launchctl load。返回 plist 路径。
    """
    agents_dir = Path.home() / "Library" / "LaunchAgents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    script_path = DATA_DIR / f"{INTRADAY_LABEL.split('.')[-1]}.sh"
    script_path.write_text(build_intraday_sh(codes), encoding="utf-8")
    script_path.chmod(
        script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    logger.info("已生成盘中信号调度脚本 %s", script_path)

    plist_path = agents_dir / f"{INTRADAY_LABEL}.plist"
    plist_path.write_text(
        build_plist(INTRADAY_LABEL, script_path, cfg.schedule_hour, cfg.schedule_minute),
        encoding="utf-8",
    )
    logger.info(
        "已生成盘中信号 LaunchAgent %s（工作日 %02d:%02d）",
        plist_path, cfg.schedule_hour, cfg.schedule_minute,
    )

    if load:
        subprocess.run(
            ["launchctl", "unload", str(plist_path)],
            capture_output=True, check=False,
        )
        subprocess.run(["launchctl", "load", str(plist_path)], check=True)
        logger.info(
            "launchctl 已加载盘中信号任务：工作日 %02d:%02d 推送",
            cfg.schedule_hour, cfg.schedule_minute,
        )
    else:
        logger.info("未加载。启用请运行: launchctl load %s（或重新执行 --load）", plist_path)
    return plist_path


# ---- 补跑守护（主任务错过时自动补推）----

# 守护检查时点：主任务 9:51 之后每 10 分钟一次，至 11:21（11:30 截止）
GUARD_TIMES = [(10, m) for m in (1, 11, 21, 31, 41, 51)] + [(11, m) for m in (1, 11, 21)]


def build_guard_sh(codes: List[str]) -> str:
    """生成补跑守护脚本：当天未推送过信号则补跑（幂等检查在 serve-guard 内）。"""
    codes_arg = ",".join(codes)
    return (
        "#!/bin/bash\n"
        f"cd {ROOT}\n"
        f"exec {ROOT}/.venv/bin/python main.py serve-guard --codes {codes_arg}\n"
    )


def build_guard_plist(label: str, script_path: Path) -> str:
    """生成守护任务 plist：工作日多个时点触发（每 10 分钟检查一次）。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    intervals = "\n".join(
        f"""        <dict>
            <key>Weekday</key>
            <integer>{wd}</integer>
            <key>Hour</key>
            <integer>{h}</integer>
            <key>Minute</key>
            <integer>{m}</integer>
        </dict>"""
        for wd in range(1, 6) for h, m in GUARD_TIMES
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


def install_guard_scheduler(codes: List[str], load: bool = False) -> Path:
    """安装补跑守护任务：主任务错过时（电脑休眠等）自动补推。"""
    agents_dir = Path.home() / "Library" / "LaunchAgents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    script_path = DATA_DIR / f"{GUARD_LABEL.split('.')[-1]}.sh"
    script_path.write_text(build_guard_sh(codes), encoding="utf-8")
    script_path.chmod(
        script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    logger.info("已生成补跑守护脚本 %s", script_path)

    plist_path = agents_dir / f"{GUARD_LABEL}.plist"
    plist_path.write_text(build_guard_plist(GUARD_LABEL, script_path), encoding="utf-8")
    logger.info("已生成补跑守护 LaunchAgent %s（工作日每 10 分钟检查）", plist_path)

    if load:
        subprocess.run(
            ["launchctl", "unload", str(plist_path)],
            capture_output=True, check=False,
        )
        subprocess.run(["launchctl", "load", str(plist_path)], check=True)
        logger.info("launchctl 已加载补跑守护：工作日 10:01~11:21 每 10 分钟检查补推")
    else:
        logger.info("未加载。启用请运行: launchctl load %s（或重新执行 --load）", plist_path)
    return plist_path


# ---- 次日预判（隔夜推送）----


def run_nextday_job(codes: List[str], dry_run: bool = False) -> List[dict]:
    """次日预判任务：逐股预测明日涨跌 → 持久化 → 推送。非交易日返回 None 跳过。"""
    from serving.nextday import format_nextday_message, predict_nextday, push_nextday
    from serving.signal_store import save_signal

    results: List[dict] = []
    for code in codes:
        try:
            sig = predict_nextday(code)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s 次日预测失败: %s", code, exc)
            continue
        if sig is None:
            logger.info("%s 跳过（数据不足）", code)
            continue
        save_signal(sig)
        if dry_run:
            print(format_nextday_message(sig))
        elif cfg.push_enabled:
            push_nextday(sig)
        results.append(sig)
    logger.info("次日预判完成：%d/%d 只产生信号", len(results), len(codes))
    return results


def build_nextday_sh(codes: List[str]) -> str:
    """生成次日预判任务 wrapper 脚本：早盘前推送明日预判。"""
    codes_arg = ",".join(codes)
    return (
        "#!/bin/bash\n"
        f"cd {ROOT}\n"
        f"exec {ROOT}/.venv/bin/python main.py nextday-push --codes {codes_arg}\n"
    )


def install_nextday_scheduler(codes: List[str], load: bool = False) -> Path:
    """安装次日预判定时任务：工作日早盘前（08:40）推送明日预判。"""
    agents_dir = Path.home() / "Library" / "LaunchAgents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    script_path = DATA_DIR / f"{NEXTDAY_LABEL.split('.')[-1]}.sh"
    script_path.write_text(build_nextday_sh(codes), encoding="utf-8")
    script_path.chmod(
        script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    logger.info("已生成次日预判脚本 %s", script_path)

    plist_path = agents_dir / f"{NEXTDAY_LABEL}.plist"
    plist_path.write_text(
        build_plist(NEXTDAY_LABEL, script_path, cfg.nextday_hour, cfg.nextday_minute),
        encoding="utf-8",
    )
    logger.info(
        "已生成次日预判 LaunchAgent %s（工作日 %02d:%02d 早盘前）",
        plist_path, cfg.nextday_hour, cfg.nextday_minute,
    )

    if load:
        subprocess.run(
            ["launchctl", "unload", str(plist_path)],
            capture_output=True, check=False,
        )
        subprocess.run(["launchctl", "load", str(plist_path)], check=True)
        logger.info(
            "launchctl 已加载次日预判：工作日 %02d:%02d 推送明日预判",
            cfg.nextday_hour, cfg.nextday_minute,
        )
    else:
        logger.info("未加载。启用请运行: launchctl load %s（或重新执行 --load）", plist_path)
    return plist_path
