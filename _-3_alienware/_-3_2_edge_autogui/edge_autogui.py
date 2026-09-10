"""Compose local queries and search in a focused Edge window using PyAutoGUI."""

import argparse
from datetime import date
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time


# -----------------------------------------------------------------------------
# Arguments and vocabulary
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=10, help="Number of random searches after daily priorities (default: 10)")
    parser.add_argument("--vocabulary", type=Path, default=Path(__file__).with_name("vocabulary.json"))
    parser.add_argument("--start-delay", type=float, default=5, help="Seconds before desktop input starts")
    parser.add_argument("--launch-wait", type=float, default=5, help="Seconds to allow Edge to open")
    parser.add_argument("--skip-launch", action="store_true", help="Use an already focused Edge window")
    parser.add_argument("--load-wait", type=float, default=4, help="Seconds to allow results to load")
    parser.add_argument("--wait-min", type=float, default=5, help="Minimum wait after scrolling")
    parser.add_argument("--wait-max", type=float, default=12, help="Maximum wait after scrolling")
    parser.add_argument("--scroll-min", type=int, default=3, help="Minimum downward scroll clicks")
    parser.add_argument("--scroll-max", type=int, default=8, help="Maximum downward scroll clicks")
    parser.add_argument("--seed", type=int, help="Optional reproducible random seed")
    parser.add_argument("--part-prob", type=float, default=0.6, help="Inclusion probability for each query group (default: 0.6)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without desktop input or waits")
    args = parser.parse_args()

    if args.repeat < 1: parser.error("--repeat must be at least 1")
    if not 0 <= args.part_prob <= 1: parser.error("--part-prob must be between 0 and 1")
    delays = (args.start_delay, args.launch_wait, args.load_wait, args.wait_min, args.wait_max)
    if any(not math.isfinite(value) or value < 0 for value in delays):
        parser.error("Delays must be finite, nonnegative numbers")
    if args.wait_max < args.wait_min: parser.error("--wait-max must be >= --wait-min")
    if args.scroll_min < 1 or args.scroll_max < args.scroll_min:
        parser.error("Scroll bounds must satisfy 1 <= --scroll-min <= --scroll-max")
    return args


def load_vocabulary(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data: raise ValueError("Vocabulary must be a nonempty JSON object")

    for name, entries in data.items():
        if not isinstance(entries, list) or not entries: raise ValueError(f"{name}: expected a nonempty list")
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f"{name}, entry {index}: expected a nonempty string, got {entry!r}")
            if not entry.isascii() or not entry.isprintable():
                raise ValueError(f"{name}, entry {index}: use printable ASCII letters, digits, punctuation, and spaces; got {entry!r}")
    return data


# -----------------------------------------------------------------------------
# X11 display and focus
# -----------------------------------------------------------------------------
def wake_screen():
    if not sys.platform.startswith("linux"): return
    print("Requesting X11 screen wake.", flush=True)
    for command in (["xset", "s", "reset"], ["xset", "dpms", "force", "on"]):
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
    time.sleep(2)


def check_edge_focus():
    if not sys.platform.startswith("linux"): return
    active = subprocess.run(["xprop", "-root", "_NET_ACTIVE_WINDOW"],
                            capture_output=True, text=True, check=True, timeout=5)
    match = re.search(r"0x[0-9a-fA-F]+", active.stdout)
    if not match or int(match.group(), 16) == 0: raise RuntimeError("No active X11 window after startup.")

    window = subprocess.run(["xprop", "-id", match.group(), "WM_CLASS"],
                            capture_output=True, text=True, check=True, timeout=5)
    classes = re.findall(r'"([^"\n]+)"', window.stdout.lower())
    if not any(name.startswith(("microsoft-edge", "msedge")) for name in classes):
        raise RuntimeError(f"Edge is not focused; stopping desktop input. Active window: {window.stdout.strip()}")


# -----------------------------------------------------------------------------
# Browser startup
# -----------------------------------------------------------------------------
def launch_edge(gui, launch_wait):
    gui.press("win")
    time.sleep(1)
    gui.write("edge", interval=0.1)
    time.sleep(1)
    if sys.platform == "win32": gui.press("enter")
    else: gui.hotkey("ctrl", "1")
    time.sleep(launch_wait)


# -----------------------------------------------------------------------------
# Query composition
# -----------------------------------------------------------------------------
def compose_query(vocabulary, rng, probability):
    groups = [entries for entries in vocabulary.values() if rng.random() < probability]
    if not groups: groups = [rng.choice(list(vocabulary.values()))]

    parts = []
    for entries in groups:
        phrase = rng.choice(entries)
        letters = [index for index, char in enumerate(phrase) if char.isalpha()]
        count = min(rng.choice((0, 1, 2)), max(0, len(letters) - 1))
        removed = set(rng.sample(letters, count))
        parts.append("".join(char for index, char in enumerate(phrase) if index not in removed))
    return " ".join(parts)


# -----------------------------------------------------------------------------
# Daily priority queries
# -----------------------------------------------------------------------------
def daily_priorities():
    queries = json.loads(Path(__file__).with_name("priority_queries.json").read_text(encoding="utf-8"))
    if not isinstance(queries, list) or any(not isinstance(q, str) or not q.strip() or
                                          not q.isascii() or not q.isprintable() for q in queries):
        raise ValueError("Priority queries must be a list of nonempty printable ASCII strings")
    path = Path.home() / ".local/state/edge-autogui/priorities.json"
    progress = json.loads(path.read_text()) if path.exists() else {}
    today = date.today().isoformat()
    completed = progress.get("completed", []) if progress.get("date") == today else []
    return [query for query in queries if query not in completed], path, {"date": today, "completed": completed}


def save_priority(path, progress, query):
    progress["completed"].append(query)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as saved:
        json.dump(progress, saved)
    Path(saved.name).replace(path)


# -----------------------------------------------------------------------------
# Query generation and desktop interaction
# -----------------------------------------------------------------------------
def run(args, vocabulary, gui=None):
    rng = random.Random(args.seed)
    submitted = 0
    priorities, progress_path, progress = daily_priorities()
    total = len(priorities) + args.repeat
    launch_key = "Enter" if sys.platform == "win32" else "Ctrl+1"
    if not args.skip_launch: print(f"Startup: Windows/Super -> edge -> {launch_key}", flush=True)
    if gui:
        print(f"Starting in {args.start_delay:g}s. Keep the pointer over Edge page content once open.", flush=True)
        if args.skip_launch: print("Focus your existing Edge window now.", flush=True)
        time.sleep(args.start_delay)
        wake_screen()
        if not args.skip_launch: launch_edge(gui, args.launch_wait)

    for index in range(total):
        priority = index < len(priorities)
        query = priorities[index] if priority else compose_query(vocabulary, rng, args.part_prob)
        scroll = rng.randint(args.scroll_min, args.scroll_max)
        wait = rng.uniform(args.wait_min, args.wait_max)
        print(f"[{index + 1}/{total}] {query} | scroll={scroll}, wait={wait:.1f}s", flush=True)
        if gui is None: continue

        check_edge_focus()
        gui.hotkey("ctrl", "l")
        gui.write(query, interval=0.06)
        gui.press("enter")
        submitted += 1
        time.sleep(args.load_wait)
        gui.scroll(-scroll)
        time.sleep(wait)
        if priority: save_priority(progress_path, progress, query)

    if gui is None:
        print(f"Would leave in address bar without Enter: I am done searching : {total} times", flush=True)
        return

    check_edge_focus()
    message = f"I am done searching : {submitted} times"
    gui.hotkey("ctrl", "l")
    gui.write(message, interval=0.06)
    print(message, flush=True)


def main():
    args = parse_args()
    try:
        vocabulary = load_vocabulary(args.vocabulary)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Vocabulary error: {error}") from error

    if args.dry_run:
        run(args, vocabulary)
        return
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        raise SystemExit("Use an X11 desktop session on Linux; native Wayland is not supported.")

    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.3
    try:
        run(args, vocabulary, pyautogui)
    except (KeyboardInterrupt, pyautogui.FailSafeException):
        raise SystemExit("Stopped before completion.") from None
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(f"Desktop automation failed: {error}") from error


if __name__ == "__main__":
    main()
