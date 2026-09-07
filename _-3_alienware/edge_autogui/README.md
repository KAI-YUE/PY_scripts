## Run

Preview generated queries and timing without importing PyAutoGUI or sending input:

```bash
python3 edge_autogui.py --repeat 5 --dry-run --seed 42
```

Run 20 searches, waiting between 5 and 15 seconds after each scroll:

```bash
python3 edge_autogui.py --repeat 20 --wait-min 5 --wait-max 15
```

After the default five-second countdown, Linux runs `xset s reset` and
`xset dpms force on`, then waits two seconds to wake a blanked display.
This does not change idle timeouts, unlock the session, or wake a suspended
laptop. Linux requires `xset` and `xprop` (`x11-xserver-utils` and `x11-utils`
on Ubuntu/Pop!_OS). A desktop lid policy may still keep the physical panel off.

The script then presses Windows/Super,
waits one second, types `edge`, waits one second, then presses `Ctrl+1`.
It waits another five seconds for Edge to open before searching; change this
with `--launch-wait 10` if needed. This uses your requested launcher shortcut:
your desktop must support `Ctrl+1` to open the matching result. It is not a
universal launcher shortcut across Windows and Linux desktops.

Use `--skip-launch` to bypass the startup sequence and focus an existing Edge
window during the countdown instead. Place the mouse pointer over Edge's page
content so scroll events reach the page. Searches use Edge's configured
address-bar search engine. Keep Edge focused while running; input goes to
whichever application has focus. On Linux, the script checks the active window's
`WM_CLASS` before each query and fails if it is not Edge. This checks focus,
not whether results loaded. Windows does not have this focus check.

Move the pointer into a corner of the primary screen to trigger PyAutoGUI's
fail-safe on its next input call, or interrupt the script with Ctrl+C in its
terminal. The fail-safe is checked after any current sleep finishes.

Other parameters: `--start-delay 8`, `--load-wait 6`, `--scroll-min 3`, and
`--scroll-max 8`. Scroll values are wheel clicks; distance depends on desktop
settings. Loading uses a fixed delay, so increase `--load-wait` for slow pages.
The final iteration also scrolls and waits. This script sends input; it does not
inspect results or assert that the search succeeded.

## Closed-lid schedule (Linux)

`edge_schedule.py` follows the `codex_ping` pattern: daily cron times, a lid check,
an overlap lock, rotating logs, and queued desktop notifications. At the scheduled
minute it checks `/proc/acpi/button/lid/*/state` once. It runs only when all lid
sensors report closed; an open lid postpones the event by one hour. A minute-by-minute cron check
runs the pending event once due; each open-lid attempt postpones another hour.
Missing or unknown lid state fails without desktop input and also retains the retry.
Retries persist in `~/.local/state/edge-autogui/retry.json` across process exits.
Overlapping scheduled events merge into one pending run, without moving its
existing deadline. Failed runs also retry after an hour from their attempt start;
success clears the pending event. Missed initial cron events during suspend are
not recreated, but an already pending retry runs after the machine resumes.

Edit the settings at the bottom of `edge_schedule.py`:

```python3
time_stamps = ["06:03"]
search_args = ["--repeat", "1", "--wait-min", "5", "--wait-max", "15"]
```

The default is **one random search for debugging**, plus unfinished daily priorities. Change `"1"` to `"20"` when ready.
For your 05:50 test, set `time_stamps = ["05:50"]` before installing.
These are daily times in cron's timezone (normally the machine's local time);
if 05:50 has passed, the next run is tomorrow. Remove or reset the schedule after
testing if you do not want it repeated tomorrow.

From a terminal inside your logged-in X11 desktop, using the Python environment
where PyAutoGUI is installed:

```bash
python3 edge_schedule.py --dry-run
python3 edge_schedule.py --install
crontab -l
```

Requires an active cron service, `crontab`, and `notify-send` (`libnotify-bin` on
Ubuntu/Pop!_OS). Run without `sudo`. Installation preserves unrelated entries,
including `codex_ping`, and saves the current desktop connection variables to
`~/.local/state/edge-autogui/desktop.json` for cron. Reinstall after logging into
a new desktop session, moving this folder, changing Python environments, or
editing schedule times. Search arguments are read from the script each run.

The laptop must **stay awake with an unlocked, usable X11 desktop while the lid
is closed**. Cron cannot wake a suspended laptop, and PyAutoGUI cannot operate
Edge through a lock screen. This script does not change power or lock settings.
The existing launcher shortcut must work, and the pointer must be over the
browser page for scrolling. Closing the lid can change your display layout;
verify it with the debug test before increasing the repeat count.

Start, completion, open-lid postponement, and failure notifications use `notify-send`.
Failed deliveries stay queued and are retried every minute. Desktop notification
settings control visibility/history. Completion means the input sequence exited
successfully, not that search results were verified. A run times out after one
hour. The desktop fail-safe exits with failure so the scheduler reports a stop
as a failed run.

```bash
tail -n 30 ~/.local/state/edge-autogui/search.log
python3 edge_schedule.py --uninstall
```

Uninstall removes this scheduler's search, retry, and notification cron entries
and cancels the pending retry. It
does not stop a currently running search or remove logs. No schedule is installed
until you run `--install`.

## Daily priority queries

Edit `priority_queries.json` for the nine exact queries. They run in file order
before random queries, with no part sampling or letter deletion. Scrolling and
random waits still apply. On the first run of a local calendar day, `--repeat 1`
therefore runs nine priorities plus one random query; `--repeat 20` runs 29 total.
Later runs that day skip priorities already completed.

Each completed priority input sequence is recorded in
`~/.local/state/edge-autogui/priorities.json`. An interrupted run resumes with the
unfinished priorities. Completion records desktop input, not verified search
results; a crash before recording completion can cause that query to repeat.
The day is determined at run start. Direct and scheduled runs share this state;
avoid running them concurrently. Dry runs preview outstanding priorities without
updating state or sending input.

After updating to this version, run `python3 edge_schedule.py --install` once
from your X11 terminal to add the retry cron entry. Your configured daily times
are preserved in `edge_schedule.py`.

## Instant launch

`search_edge.sh` runs immediately in your current desktop session, without a lid
check or startup countdown. It defaults to 20 random queries after outstanding
daily priorities, with 5–15 second waits. Screen-wake and launcher delays still
apply. It shares the scheduler's run lock to prevent overlapping input. It uses
`python3` on your current PATH and requires `flock` (util-linux). It does not
change cron, clear pending retries, or send scheduler notifications.

The `search_edge` alias is registered in `~/.bash_aliases` on this machine.
Open a new Bash terminal or load it into your current terminal:

```bash
source ~/.bash_aliases
search_edge
search_edge --repeat 1
search_edge --dry-run --repeat 1
```

Extra arguments override the launcher's defaults. You can also run
`./search_edge.sh` from this folder. Daily priority progress is shared with cron.

After a successful run, the address bar contains
`I am done searching : N times`, where N counts priority and random queries
submitted during that run. This final message is typed without pressing Enter,
so it adds no search and the last results page remains loaded. The count records
submitted input, not verified results. Dry runs preview the planned final message.

## Windows instant launch

Copy this folder to Windows and install Python 3 with `python` available on PATH.
From a terminal in the copied folder:

```bat
python -m pip install -r requirements.txt
search_edge.bat --repeat 1
```

In PowerShell, use `./search_edge.bat --repeat 1`. Double-clicking the batch file
uses the default 20 random searches plus outstanding daily priorities. Startup
on Windows presses the Windows key, types `edge`, then presses Enter. Linux
continues to use Ctrl+1. Both use the same vocabulary and final count message.

Use `search_edge.bat --dry-run --repeat 1` to preview without desktop input.
The Windows launcher does not provide the Linux scheduler, lid detection,
notifications, or overlap lock; run only one instance at a time. Keep the desktop
unlocked and Edge focused. Linux-only display wake and focus checks are skipped.
