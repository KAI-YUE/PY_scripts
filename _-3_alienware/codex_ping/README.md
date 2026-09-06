# Codex lid ping

Runs a small, separate Codex request at 06:00 and 14:00 daily when the laptop
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

From the `_5_PYscripts/_-3_alienware/codex_ping` directory:

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

- Reads `/proc/acpi/button/lid/*/state`; an open or unknown lid prevents a request.
- Uses an empty temporary working directory, a read-only sandbox, and an ephemeral
  session. User config is ignored for the request, while saved authentication is reused.
- Sets `project_doc_max_bytes=0` for the ping to disable loading global/project
  `AGENTS.md` instructions. The empty working directory also keeps project files
  out of the request. Your normal VS Code instructions and settings are unchanged.
- Sends `Reply only OK. Do not use tools, read files, or perform any other work.`
- Prevents overlapping requests and limits the request to 90 seconds.
- Logs skips, success, and failures to `~/.local/state/codex-ping/ping.log`, with
  two rotated backups. It does not log authentication tokens or response content.
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

Official references: [scripted Codex runs](https://learn.chatgpt.com/docs/non-interactive-mode)
and [usage limits](https://learn.chatgpt.com/docs/pricing).
