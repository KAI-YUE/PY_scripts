# Codex lid ping

Runs a small, separate Codex request at 03:50, 09:00, 11:05, 14:00 and 18:30 daily when the laptop
lid is closed. The laptop must remain awake and have internet access.

Cron stores a schedule in your user's crontab and launches this Python script.
There is no `.dat` file to send to VS Code, and no message is inserted into an
existing VS Code conversation. Codex CLI requests made with the same ChatGPT
account use that account's allowance. Whether a request starts the desired
five-hour window must be checked against your account's reset time; this script
does not reset limits or guarantee their timing.

## Setup on this machine or a new Linux machine

Requires Python 3, `crontab`, an active cron service, `notify-send` (the
`libnotify-bin` package on Pop!_OS), and either Codex CLI on PATH
or the Codex VS Code extension under `~/.vscode/extensions`. No pip packages needed.
Run as your normal desktop user, without `sudo`.

From the `_5_PYscripts` project directory:

```bash
sudo apt install libnotify-bin
cd ./_-3_alienware/codex_ping/
python3 codex_ping.py --dry-run
python3 codex_ping.py --install
crontab -l
```

The installer preserves other cron entries and replaces its own marked entries.
Run `--install` again after updating the script to add the notification check.
The times use the cron service's timezone (normally the machine's local timezone).
If your crontab already sets `CRON_TZ`, that may affect scheduling.
After copying the project to another machine or moving it, run `--install` again
to record the new absolute script path. The script must remain at that path.

The script finds `codex` on PATH first, then the newest installed VS Code Codex
binary by modification time, so extension version paths are not hard-coded.
If authentication fails, run `codex login` with your ChatGPT account. If `codex`
is not on your terminal PATH, use the executable path printed by `--dry-run`.
API-key login is intentionally rejected because API usage is billed separately.
Cron needs access to the same home directory and saved login as your desktop user.
If you use a custom `CODEX_HOME`, set it explicitly in your crontab.

## Behavior and checks

To change the daily schedule, edit `time_stamps` in the `__main__` block at the
bottom of `codex_ping.py`, then run `python3 codex_ping.py --install` from its folder:

```python
time_stamps = ["06:00", "14:00", "19:30"]
```

Use local 24-hour `HH:MM` times. Add or remove entries as needed; duplicates are
ignored and invalid times leave the installed schedule untouched. An empty list
disables scheduled pings while keeping notification retries. Use `--uninstall`
to remove both. Editing the list alone does not update cron.

- Reads `/proc/acpi/button/lid/*/state`; an open or unknown lid prevents a request.
- Before each request, logs NetworkManager device states. If neither Wi-Fi nor
  Ethernet is connected, enables Wi-Fi and waits up to 20 seconds for saved
  autoconnect profiles. This cannot override a hardware radio block, missing
  credentials, or NetworkManager permission restrictions in cron.
- Logs a ping to `1.1.1.1` and an HTTPS probe to `https://chatgpt.com/`, including
  DNS/TCP/TLS timings. Probe failures do not prevent the Codex attempt; ICMP may
  be blocked and HTTP 403 still shows that an HTTPS server was reached.
  Requires `nmcli`, `nm-online`, `ping`, and `curl` for these checks; missing
  utilities are logged. Connected interfaces are not restarted.
- Uses an empty temporary working directory, a read-only sandbox, and an ephemeral
  session. User config is ignored for the request, while saved authentication is reused.
- Sets `project_doc_max_bytes=0` for the ping to disable loading global/project
  `AGENTS.md` instructions. The empty working directory also keeps project files
  out of the request. Your normal VS Code instructions and settings are unchanged.
- Sends `Reply only OK. Do not use tools, read files, or perform any other work.`
- Prevents overlapping requests and limits the request to 90 seconds.
- Requires JSON events confirming a completed turn, an `OK` reply, and positive
  input/output token counts. Exit code zero alone does not count as success.
- Logs skips, success, and failures to `~/.local/state/codex-ping/ping.log`, with
  two rotated backups. Verified runs include input/output token counts; raw
  output is normally omitted. On timeout, nonzero exit, or response verification failure, the last 4,000
  characters of each output stream are logged with common credential patterns
  redacted. Review diagnostics before sharing them.
- Logs the selected CLI path/version and start times and elapsed durations for
  network checks, individual probes, login checks, and the Codex request, including
  elapsed duration when a command fails or times out.
- Does not retry failures; the next attempt is the next scheduled run.
- Queues success/failure messages as `.pending` files in the log directory and
  immediately sends them through the desktop notification service, regardless of
  lid state. Delivered messages are removed from the queue. A second cron entry
  runs `--notify` every minute to retry failed deliveries; it never calls Codex.
  Open-lid skips stay in the log.
- Requests a notification without an expiry, but the desktop controls its actual
  display and history (including Do Not Disturb). Delivery is attempted as soon as
  the request finishes. Failed deliveries stay queued for another attempt.
  Notifications use the current user's desktop session bus under `/run/user/UID`.

Dry run checks the lid and executable without calling the model. To run normally
right now, omit `--dry-run`; this consumes usage if the lid is closed.

```bash
tail -n 20 ~/.local/state/codex-ping/ping.log
python3 codex_ping.py --uninstall
```

After the first successful request following an idle period, check the Codex
usage dashboard or CLI `/status` to see whether the reset time matches your goal.
Notifications confirm a model response, not that a new five-hour window started.
Earlier scheduled or manual activity can still affect the window. Cron does not wake a
suspended laptop. Reinstall the schedule after changing the times.

Official references: [scripted Codex runs](https://learn.chatgpt.com/docs/non-interactive-mode)
and [usage limits](https://learn.chatgpt.com/docs/pricing).
