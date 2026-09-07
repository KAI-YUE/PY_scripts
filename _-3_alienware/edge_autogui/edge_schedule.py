#!/usr/bin/env python3
"""Run Edge desktop searches when the laptop lid is closed (Linux)."""

import argparse
import json
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
import time

MARKER = "# edge-lid-search"
STATE = Path.home() / ".local/state/edge-autogui"


# --- Helper: lid_closed
def lid_closed():
    sensors = list(Path("/proc/acpi/button/lid").glob("*/state"))
    if not sensors:
        raise RuntimeError("No ACPI lid sensor found; skipping the searches.")
    states = [p.read_text().strip().removeprefix("state:").strip() for p in sensors]
    if any(state not in ("open", "closed") for state in states):
        raise RuntimeError("Unknown lid state; skipping the searches.")
    return all(state == "closed" for state in states)


# --- Helper: queue_notification
def queue_notification(message):
    STATE.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=STATE, delete=False) as pending:
        pending.write(f"{datetime.now():%Y-%m-%d %H:%M}: {message}")
    Path(pending.name).rename(pending.name + ".pending")
    try:
        notify_pending()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
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
            subprocess.run([executable, "--app-name=Edge searches", "--expire-time=0",
                            "Edge scheduled searches", pending.read_text()],
                           env=env, capture_output=True, timeout=15, check=True)
            pending.unlink()
            logging.info("Desktop notification delivered.")


# --- Helper: update_schedule
def update_schedule(time_stamps, remove=False):
    times = []
    if not remove:
        for stamp in time_stamps:
            try:
                time = datetime.strptime(stamp, "%H:%M")
            except (TypeError, ValueError):
                raise RuntimeError(f"Invalid time stamp {stamp!r}; use HH:MM (00:00–23:59).") from None
            times.append((time.hour, time.minute))
        times = sorted(set(times))

    if not remove:
        if os.environ.get("XDG_SESSION_TYPE", "").lower() != "x11" or not os.environ.get("DISPLAY"):
            raise RuntimeError("Install from a terminal inside your X11 desktop session.")
        desktop_env = {key: os.environ[key] for key in
                       ("DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR", "XDG_SESSION_TYPE")
                       if key in os.environ}
        STATE.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=STATE, delete=False) as saved:
            json.dump(desktop_env, saved)
        Path(saved.name).replace(STATE / "desktop.json")

    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if current.returncode and "no crontab for" not in current.stderr.lower():
        raise RuntimeError(current.stderr.strip() or "Cannot read crontab.")
    lines = [line for line in current.stdout.splitlines() if not line.endswith(MARKER)]

    if not remove:
        command = shlex.join([sys.executable, str(Path(__file__).resolve())])
        if "\n" in command or "\r" in command:
            raise RuntimeError("The script path must not contain a newline.")
        command = command.replace("%", r"\%")
        for hour, minute in times:
            lines.append(f"{minute} {hour} * * * {command} {MARKER}")
        lines.append(f"* * * * * {command} --notify {MARKER}")
        lines.append(f"* * * * * {command} --retry {MARKER}")

    subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True, check=True)
    if remove: (STATE / "retry.json").unlink(missing_ok=True)
    schedule = ", ".join(f"{hour:02d}:{minute:02d}" for hour, minute in times) or "no searches"
    print("Schedule removed." if remove else
          f"Installed: {schedule}; queued notifications checked every minute.")


# --- Helper: search
def search(search_args, dry_run=False, retry=False):
    command = [sys.executable, "-u", str(Path(__file__).with_name("edge_autogui.py")), *search_args]
    if dry_run:
        closed = lid_closed()
        print(f"Lid: {'closed' if closed else 'open'}")
        print(f"Command: {shlex.join(command)}")
        print(f"Saved desktop environment: {(STATE / 'desktop.json').exists()}")
        print(f"Notifications: {shutil.which('notify-send') or 'missing; install libnotify-bin'}")
        print("Would run searches." if closed else "Would postpone one hour: lid is open.")
        return

    with (STATE / "search.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logging.info("Skipped: another search run is active.")
            return

        pending = STATE / "retry.json"
        if retry and not pending.exists(): return
        if pending.exists() and time.time() < json.loads(pending.read_text())["due"]: return

        # Persist the next attempt before checking the lid or starting the child.
        due = time.time() + 3600
        with tempfile.NamedTemporaryFile(mode="w", dir=STATE, delete=False) as saved:
            json.dump({"due": due}, saved)
        Path(saved.name).replace(pending)
        if not lid_closed():
            message = f"Lid is open; searches postponed until {datetime.fromtimestamp(due):%Y-%m-%d %H:%M}."
            logging.info(message)
            queue_notification(message)
            return

        env = os.environ.copy()
        env.update(json.loads((STATE / "desktop.json").read_text()))
        logging.info("Starting: %s", shlex.join(command))
        queue_notification("Starting scheduled Edge searches.")
        result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=3600)
        logging.info("Search output:\n%s%s", result.stdout, result.stderr)
        if result.returncode: raise RuntimeError(f"Search script failed (exit {result.returncode}).")
        pending.unlink(missing_ok=True)
        logging.info("Desktop input sequence completed; search results were not verified.")
        queue_notification("Edge input sequence completed. Search results were not verified.")




# -----------------------------
# main
# -----------------------------
def main(time_stamps, search_args):
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--dry-run", action="store_true", help="Check lid and command without desktop input")
    options.add_argument("--install", action="store_true", help="Install/update the daily cron entry")
    options.add_argument("--uninstall", action="store_true", help="Remove this script's cron entry")
    options.add_argument("--notify", action="store_true", help="Retry delivery of queued notifications")
    options.add_argument("--retry", action="store_true", help="Run a postponed event when due")
    args = parser.parse_args()
    try:
        if not (args.dry_run or args.install or args.uninstall):
            STATE.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(STATE / "search.log", maxBytes=262144, backupCount=2)
            logging.basicConfig(level=logging.INFO, handlers=[handler],
                                format="%(asctime)s %(levelname)s %(message)s")

        if args.install or args.uninstall:
            update_schedule(time_stamps, remove=args.uninstall)
        elif args.notify:
            notify_pending()
        else:
            search(search_args, dry_run=args.dry_run, retry=args.retry)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        logging.error("%s", error)
        if not (args.dry_run or args.install or args.uninstall or args.notify):
            try:
                queue_notification("Searches failed. See ~/.local/state/edge-autogui/search.log for details.")
            except OSError as queue_error:
                logging.error("Cannot queue notification: %s", queue_error)
        return 1
    return 0


if __name__ == "__main__":
    # Daily local times (24-hour HH:MM); edit this list, then run --install again.
    time_stamps = ["04:06"]
    # Debug: one search per run. Change to "20" for the full loop.
    search_args = ["--repeat", "1", "--wait-min", "5", "--wait-max", "42"]
    sys.exit(main(time_stamps, search_args))
