#!/usr/bin/env python3
"""Watchdog for KMURecSys26 Steam aggressive quota runner.

Silent when the runner is healthy or the deadline has passed. If the runner is missing
before the deadline, restart it detached and print one concise notification.
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/opt/data/kaggle/kmu-rec-sys-26-steam')
SCRIPT = ROOT / 'scripts/aggressive_quota_runner.py'
LOG_DIR = ROOT / 'logs'
KST = ZoneInfo('Asia/Seoul')
DEADLINE = dt.datetime(2026, 6, 15, 23, 59, 59, tzinfo=KST)


def runner_pids() -> list[int]:
    pids: list[int] = []
    self_pid = os.getpid()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid == self_pid:
            continue
        try:
            cmd = (p / 'cmdline').read_bytes().replace(b'\x00', b' ').decode('utf-8', 'ignore')
        except Exception:
            continue
        if 'scripts/aggressive_quota_runner.py' in cmd and 'uv run' in cmd or str(SCRIPT) in cmd:
            pids.append(pid)
    return sorted(set(pids))


def main() -> None:
    now = dt.datetime.now(KST)
    if now > DEADLINE:
        return
    if runner_pids():
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime('%Y%m%dT%H%M%SKST')
    log_path = LOG_DIR / f'aggressive_quota_runner_watchdog_{stamp}.log'
    env = os.environ.copy()
    env.update({
        'HOME': '/opt/data/home',
        'WANDB_ENTITY': 'mrpc2003-kookmin-university',
        'WANDB_PROJECT': 'kmu-rec-sys-26-steam',
        'PYTHONUNBUFFERED': '1',
        'UV_NO_ACTIVE_VENV': '1',
    })
    env.pop('VIRTUAL_ENV', None)
    cmd = [
        'uv', 'run', '--with', 'pandas', '--with', 'numpy', '--with', 'scipy', '--with', 'wandb',
        'python', str(SCRIPT), '--sleep-no-quota', '300', '--sleep-no-candidate', '600'
    ]
    with log_path.open('ab') as f:
        subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    print(f'WATCHDOG_RESTARTED aggressive quota runner; log={log_path}')


if __name__ == '__main__':
    main()
