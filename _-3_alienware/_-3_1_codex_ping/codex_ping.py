#!/usr/bin/env python3
"""Send a small Codex request when the laptop lid is closed."""

import argparse
from datetime import datetime
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from time import monotonic

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

    subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True, check=True)
    schedule = ", ".join(f"{hour:02d}:{minute:02d}" for hour, minute in times) or "no pings"
    print("Schedule removed." if remove else
          f"Installed: {schedule}; queued notifications checked every minute.")


# --- Helper: log_diagnostics
def log_diagnostics(label, output):
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    output = output or "(no output)"
    output = re.sub(r"(?i)(bearer\s+|(?:token|api[_-]?key|authorization)[\"\s:=]+)\S+",
                    r"\1[redacted]", output)
    output = re.sub(r"\b(?:sk-[\w-]+|eyJ[\w.-]+)", "[redacted]", output)
    logging.info("%s: %s", label, output[-4000:].strip().replace("\n", " | "))


# --- Helper: timed_run
def timed_run(label, command, **kwargs):
    started = monotonic()
    logging.info("%s started (timeout=%ss).", label, kwargs.get("timeout"))
    try:
        result = subprocess.run(command, **kwargs)
        logging.info("%s exited with code %s.", label, result.returncode)
        return result
    finally:
        logging.info("%s elapsed: %.2fs.", label, monotonic() - started)


# --- Helper: network_command
def network_command(label, command, timeout):
    try:
        result = timed_run(label, command, capture_output=True, text=True, timeout=timeout,
                                stdin=subprocess.DEVNULL, env={**os.environ, "LC_ALL": "C"})
        log_diagnostics(f"{label} (exit {result.returncode})", result.stdout + result.stderr)
        return result
    except (OSError, subprocess.TimeoutExpired) as error:
        logging.warning("%s: %s", label, error)
        return None


# --- Helper: prepare_network
def prepare_network():
    nmcli = shutil.which("nmcli")
    if nmcli:
        status = network_command("Network devices", [nmcli, "-t", "-f", "TYPE,STATE", "device"], 10)
        connected = status and any(line in ("wifi:connected", "ethernet:connected")
                                   for line in status.stdout.splitlines())
        if status and status.returncode == 0 and not connected:
            network_command("Enable Wi-Fi", [nmcli, "--wait", "10", "radio", "wifi", "on"], 15)
            # NetworkManager reconnects saved profiles with autoconnect enabled.
            network_command("Wait for connection", ["nm-online", "--quiet", "--timeout=20"], 25)
            network_command("Network devices after wake", [nmcli, "-t", "-f", "TYPE,STATE", "device"], 10)
    else:
        logging.warning("nmcli missing; Wi-Fi wake unavailable.")

    network_command("Internet ping", ["ping", "-n", "-c", "1", "-W", "5", "1.1.1.1"], 8)
    # ICMP can be blocked; HTTPS checks DNS, TCP and TLS on the service host too.
    network_command("ChatGPT HTTPS", ["curl", "--silent", "--show-error", "--output", "/dev/null",
                    "--connect-timeout", "5", "--max-time", "15", "--write-out",
                    "http=%{http_code} dns=%{time_namelookup}s connect=%{time_connect}s "
                    "tls=%{time_appconnect}s total=%{time_total}s\n", "https://chatgpt.com/"], 20)
    logging.info("Network checks complete; attempting Codex regardless of probe results.")


# --- Helper: verify_response
def verify_response(output):
    try:
        events = [json.loads(line) for line in output.splitlines() if line.strip()]
    except json.JSONDecodeError:
        raise RuntimeError("Codex returned invalid JSON; request is not verified.") from None
    if any(not isinstance(event, dict) for event in events):
        raise RuntimeError("Codex returned an invalid event; request is not verified.")
    if any(event.get("type") == "turn.failed" for event in events):
        raise RuntimeError("Codex reported a failed turn; request is not verified.")

    replies = [(event["item"]["text"].strip() if isinstance(event["item"].get("text"), str) else "") for event in events
               if event.get("type") == "item.completed"
               and isinstance(event.get("item"), dict)
               and event["item"].get("type") == "agent_message"]
    completed = [event for event in events if event.get("type") == "turn.completed"]
    usage = completed[-1].get("usage") if completed else None
    if not replies or replies[-1] != "OK" or not isinstance(usage, dict):
        raise RuntimeError("Codex did not complete with OK and token usage; request is not verified.")
    if any(type(usage.get(key)) is not int or usage[key] <= 0
           for key in ("input_tokens", "output_tokens")):
        raise RuntimeError("Codex reported no input/output token usage; request is not verified.")
    return usage


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

        logging.info("Scheduled ping started; Codex executable: %s", executable)
        version = timed_run("Codex version", [executable, "--version"],
                            capture_output=True, text=True, timeout=15)
        log_diagnostics("Codex version", version.stdout + version.stderr)
        started = monotonic()
        logging.info("Network checks started.")
        try:
            prepare_network()
        finally:
            logging.info("Network checks elapsed: %.2fs.", monotonic() - started)

        env = os.environ.copy()
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY"):
            env.pop(key, None)
        auth = timed_run("Codex login check", [executable, "login", "status"], env=env,
                              capture_output=True, text=True, timeout=15)
        if auth.returncode or "ChatGPT" not in auth.stdout + auth.stderr:
            raise RuntimeError("Codex must be logged in using ChatGPT; run codex login.")

        with tempfile.TemporaryDirectory(prefix="codex-ping-") as directory:
            command = [executable, "exec", "--ignore-user-config", "--ephemeral", "--json",
                       "-c", "project_doc_max_bytes=0",
                       "--sandbox", "read-only", "--skip-git-repo-check", "--cd", directory,
                       "Reply only OK. Do not use tools, read files, or perform any other work."]
            try:
                result = timed_run("Codex request", command, env=env, stdin=subprocess.DEVNULL,
                                        capture_output=True, text=True, timeout=90)
            except subprocess.TimeoutExpired as error:
                log_diagnostics("Codex timeout stdout", error.stdout)
                log_diagnostics("Codex timeout stderr", error.stderr)
                raise RuntimeError("Codex timed out after 90 seconds; see diagnostics above.") from None
        if result.returncode:
            log_diagnostics("Codex failure stdout", result.stdout)
            log_diagnostics("Codex failure stderr", result.stderr)
            raise RuntimeError(f"Codex request failed (exit {result.returncode}).")
        try:
            usage = verify_response(result.stdout)
        except RuntimeError:
            log_diagnostics("Codex verification failure stdout", result.stdout)
            log_diagnostics("Codex verification failure stderr", result.stderr)
            raise
        logging.info("Model response verified: OK; input_tokens=%s output_tokens=%s; reset time unverified.",
                     usage["input_tokens"], usage["output_tokens"])
        queue_notification("Model replied OK with token usage. Five-hour reset time is NOT verified; "
                           "check Codex usage.")


# -----------------------------
# main
# -----------------------------
def main(time_stamps):
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
            update_schedule(time_stamps, remove=args.uninstall)
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
    # Daily local times (24-hour HH:MM); edit this list, then run --install again.
    time_stamps = ["03:50", "09:00", "11:05", "14:00", "18:30"]
    sys.exit(main(time_stamps))
