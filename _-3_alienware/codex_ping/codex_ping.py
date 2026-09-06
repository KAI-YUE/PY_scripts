#!/usr/bin/env python3
"""Send a small Codex request when the laptop lid is closed."""

import argparse
from datetime import datetime
import fcntl
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

MARKER = "# codex-lid-ping"
STATE = Path.home() / ".local/state/codex-ping"


# --- Helper: find_codex
def find_codex():
    executable = shutil.which("codex")
    if executable:
        return executable

    extensions = Path.home() / ".vscode/extensions"
    candidates = [p for p in extensions.glob("openai.chatgpt-*/bin/linux-*/codex")
                  if p.is_file() and os.access(p, os.X_OK)]
    if not candidates:
        raise RuntimeError("Install the Codex CLI or the VS Code Codex extension first.")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


# --- Helper: lid_closed
def lid_closed():
    sensors = list(Path("/proc/acpi/button/lid").glob("*/state"))
    if not sensors:
        raise RuntimeError("No ACPI lid sensor found; skipping the request.")
    states = [p.read_text().strip().split()[-1] for p in sensors]
    if any(state not in ("open", "closed") for state in states):
        raise RuntimeError("Unknown lid state; skipping the request.")
    return all(state == "closed" for state in states)


# --- Helper: queue_notification
def queue_notification(message):
    STATE.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=STATE, delete=False) as pending:
        pending.write(f"{datetime.now():%Y-%m-%d %H:%M}: {message}")
    Path(pending.name).rename(pending.name + ".pending")
    try:
        notify_pending()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        logging.warning("Notification remains queued: %s", error)


# --- Helper: notify_pending
def notify_pending():
    if not list(STATE.glob("*.pending")):
        return
    executable = shutil.which("notify-send")
    if not executable:
        raise RuntimeError("Install libnotify-bin for desktop notifications; result remains queued.")

    env = os.environ.copy()
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")
    with (STATE / "notify.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return

        for pending in sorted(STATE.glob("*.pending")):
            subprocess.run([executable, "--app-name=Codex ping", "--expire-time=0",
                            "Codex scheduled ping", pending.read_text()],
                           env=env, capture_output=True, timeout=15, check=True)
            pending.unlink()
            logging.info("Desktop notification delivered.")


# --- Helper: update_schedule
def update_schedule(remove=False):
    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if current.returncode and "no crontab for" not in current.stderr.lower():
        raise RuntimeError(current.stderr.strip() or "Cannot read crontab.")
    lines = [line for line in current.stdout.splitlines() if not line.endswith(MARKER)]

    if not remove:
        command = shlex.join([sys.executable, str(Path(__file__).resolve())])
        if "\n" in command or "\r" in command:
            raise RuntimeError("The script path must not contain a newline.")
        command = command.replace("%", r"\%")
        lines.append(f"0 6,14 * * * {command} {MARKER}")
        lines.append(f"* * * * * {command} --notify {MARKER}")

    subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True, check=True)
    print("Schedule removed." if remove else
          "Installed: daily at 06:00 and 14:00; queued notifications checked every minute.")


# --- Helper: ping
def ping(dry_run=False):
    closed = lid_closed()
    if dry_run:
        executable = find_codex()
        print(f"Lid: {'closed' if closed else 'open'}\nCodex: {executable}")
        print(f"Notifications: {shutil.which('notify-send') or 'missing; install libnotify-bin'}")
        print("Would send a request." if closed else "Would skip: lid is open.")
        return

    if not closed:
        logging.info("Skipped: lid is open.")
        return

    executable = find_codex()
    with (STATE / "ping.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logging.info("Skipped: another ping is running.")
            return

        env = os.environ.copy()
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY"):
            env.pop(key, None)
        auth = subprocess.run([executable, "login", "status"], env=env,
                              capture_output=True, text=True, timeout=15)
        if auth.returncode or "ChatGPT" not in auth.stdout + auth.stderr:
            raise RuntimeError("Codex must be logged in using ChatGPT; run codex login.")

        with tempfile.TemporaryDirectory(prefix="codex-ping-") as directory:
            command = [executable, "exec", "--ignore-user-config", "--ephemeral",
                       "--sandbox", "read-only", "--skip-git-repo-check", "--cd", directory,
                       "Reply only OK. Do not use tools, read files, or perform any other work."]
            result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    timeout=90)
        if result.returncode:
            raise RuntimeError(f"Codex request failed (exit {result.returncode}).")
        logging.info("Request succeeded; check Codex usage to verify the reset time.")
        queue_notification("Request succeeded. Check Codex usage for the reset time.")


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--dry-run", action="store_true", help="Check setup without sending a request")
    options.add_argument("--install", action="store_true", help="Install/update the daily cron entry")
    options.add_argument("--uninstall", action="store_true", help="Remove this script's cron entry")
    options.add_argument("--notify", action="store_true", help="Retry delivery of queued notifications")
    args = parser.parse_args()
    try:
        if not (args.dry_run or args.install or args.uninstall):
            STATE.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(STATE / "ping.log", maxBytes=262144, backupCount=2)
            logging.basicConfig(level=logging.INFO, handlers=[handler],
                                format="%(asctime)s %(levelname)s %(message)s")

        if args.install or args.uninstall:
            update_schedule(remove=args.uninstall)
        elif args.notify:
            notify_pending()
        else:
            ping(dry_run=args.dry_run)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        logging.error("%s", error)
        if not (args.dry_run or args.install or args.uninstall or args.notify):
            try:
                queue_notification("Request failed. See ~/.local/state/codex-ping/ping.log for details.")
            except OSError as queue_error:
                logging.error("Cannot queue notification: %s", queue_error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
