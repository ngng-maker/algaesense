# Setting up Hermes Agent, your Anthropic key, and the Slack app

This is the walkthrough for the two things only you can do (I have no way to create a Slack app or hold your API key myself): installing Hermes, wiring in your Anthropic key, and creating the Slack app/bot token. Do this on whichever machine will actually run Hermes (the "brain" server per the architecture — could be this laptop, could be somewhere else; it does not need to be the Raspberry Pi).

**A note on secrets:** don't paste your Anthropic API key or Slack tokens into a chat with me. Every step below has you run a command yourself in your own terminal, where the secret goes straight into Hermes's own config file (`~/.hermes/.env` on Linux/macOS) and never passes through this conversation.

**On Windows specifically**, Hermes's real config directory is `%LOCALAPPDATA%\hermes\` (i.e. `C:\Users\<you>\AppData\Local\hermes\`), not `~/.hermes/` — the paths below use the Unix convention since that's Hermes's own documented default, but if you're on Windows, that's where `config.yaml`, `.env`, and `logs/` actually live. Confirm the exact path yourself with `hermes doctor` rather than assuming.

## 1. Install Hermes Agent

Windows (PowerShell):
```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

Or, since you're already in a Python environment for this project:
```
pip install hermes-agent
```

## 2. Connect your Anthropic key

```
hermes config set ANTHROPIC_API_KEY sk-ant-...
```

This writes the key to `~/.hermes/.env`, not `config.yaml` (secrets and settings are kept separate). Then:

```
hermes model
```

and select your Claude model from the interactive picker (e.g. Claude Sonnet 5, in `provider/model` format like `anthropic/claude-sonnet-5`).

## 3. Register this project's six MCP servers

Install `algaesense-agent` into the same Python environment Hermes runs in:

```
pip install -e packages/algaesense-agent
```

Then merge the `mcp_servers:` block from [`packages/algaesense-agent/src/algaesense_agent/profile/hermes_config.example.yaml`](../packages/algaesense-agent/src/algaesense_agent/profile/hermes_config.example.yaml) into your own `config.yaml`, filling in the real `ALGAESENSE_DATA_DIR`/`ALGAESENSE_EDGE_BASE_URL`/`ALGAESENSE_LABWIKI_ROOT` paths for your setup (and, if you want the "start a new experiment run" / remote-restart features, the `ALGAESENSE_PI_*`/`ALGAESENSE_DASHBOARD_URL` vars too — see `docs/remote_experiment_control.md`).

**Important**: Hermes only passes an MCP server the environment variables **explicitly listed under that specific server's own `env:` block** in `config.yaml` — not your OS environment, and not variables set for a different server in the same file, even one right above it. If a tool fails with a config-looking error (e.g. "needs either private_key_path or password"), check that the variable it needs is actually listed under *that* server's `env:` block, not just set somewhere else.

**Windows YAML gotcha**: any path value containing backslashes (`C:\Users\...`) must be written in **single** quotes in `config.yaml`, not double quotes or unquoted — double-quoted YAML strings process backslash escapes (`\U` gets read as the start of a Unicode escape sequence and breaks the parse), while single-quoted strings are taken completely literally.

## 4. Create the Slack app

The fastest path is Hermes's own manifest generator, which produces a ready-to-paste app definition with the right scopes and event subscriptions already filled in:

```
hermes slack manifest --agent-view --write
```

Then:
1. Go to https://api.slack.com/apps → **Create New App** → **From an app manifest**.
2. Select your workspace, paste the generated JSON, review, and create.

If you'd rather do it by hand instead of the generator, the app needs:

- **Bot Token Scopes** (Features → OAuth & Permissions): `chat:write`, `app_mentions:read`, `channels:history`, `channels:read`, `groups:history`, `groups:read`, `im:history`, `im:read`, `im:write`, `mpim:history`, `mpim:read`, `users:read`, `files:read`, `files:write`.
- **Event Subscriptions**: `message.im`, `message.mpim`, `message.channels`, `message.groups`, `app_mention`. Without `message.channels`/`message.groups` specifically, the bot never receives ordinary channel messages at all.
- **Socket Mode**: enable it (Settings → Socket Mode), then create an App-Level Token with the `connections:write` scope — this lets Hermes receive Slack events without needing a public HTTPS endpoint.

## 5. Install the app and collect your tokens

1. **Settings → Install App → Install to Workspace → Allow.**
2. Copy the **Bot User OAuth Token** (starts `xoxb-`).
3. Copy the **App-Level Token** from the Socket Mode step (starts `xapp-`).
4. Find your own Slack **Member ID**: click your profile → View full profile → ⋮ → Copy member ID (format `U0123ABC`). This goes into `SLACK_ALLOWED_USERS` — **without it set, Hermes denies every message by default**, as a safety default, not a bug.
5. Find the **Channel ID** you want the bot in: right-click the channel → View channel details → scroll to the bottom (`C...` for public, `G...` for private).

Set these (again, via Hermes's own commands so they land in `~/.hermes/.env`, not typed into chat with me):

```
hermes config set SLACK_BOT_TOKEN xoxb-...
hermes config set SLACK_APP_TOKEN xapp-...
hermes config set SLACK_ALLOWED_USERS U0123ABC
hermes config set SLACK_HOME_CHANNEL C0123ABC
```

(`SLACK_HOME_CHANNEL` is where scheduled/cron results post — set it to the same channel you'll be chatting in, or a dedicated one.)

## 6. Point the system prompt at this project's persona

Set Hermes's system prompt (channel-specific config, or `--ephemeral-system-prompt` for a quick test) to the contents of [`packages/algaesense-agent/src/algaesense_agent/profile/system_prompt.md`](../packages/algaesense-agent/src/algaesense_agent/profile/system_prompt.md) — this is what carries the propose-before-apply safety rule.

## 7. Invite the bot and start the gateway

In Slack, in the channel you want it in:
```
/invite @Hermes Agent
```

Then start the gateway:
```
hermes gateway
```

## 8. Register the weekly labwiki-consistency cron job

Once the gateway's running, in a Hermes chat session (see [`weekly_cron_job.md`](../packages/algaesense-agent/src/algaesense_agent/profile/weekly_cron_job.md) for the exact command).

## 9. Verify

- Message the bot in Slack: it should respond, and `mcp` server tools (fit/suggest/propose/apply/ingest/query/plot) should be visible to it (Hermes's own tool-listing command, e.g. `/mcp list`, confirms this).
- Ask it something read-only first (e.g. "fit a model for campaign X") before ever asking it to touch the LED, to confirm the pipeline side works before testing the actuator side.
- Confirm the propose-then-confirm rule actually holds: ask it to change the LED, and check it asks for your explicit yes before calling `apply_led_change` — if it doesn't, stop and let me know before proceeding further, since that's the one safety property this whole design depends on.

## Troubleshooting

Real problems hit during this project's first live bring-up, in the order you're likely to hit them:

### A tool fails with a config-looking error even though you set the env var

Check *which server's* `env:` block in `config.yaml` actually has it. Hermes doesn't inherit your OS environment or share variables between servers — each `mcp_servers:` entry only sees what's listed directly under its own `env:` key. Setting `ALGAESENSE_PI_HOST` permanently in Windows (`[System.Environment]::SetEnvironmentVariable(...)`) does **not** make it visible to `algaesense_actuators` unless it's also listed in that server's own block in `config.yaml`.

### Gateway starts but shows only the banner, then every MCP server logs `CancelledError`

Check `logs/errors.log` and `logs/gateway.log` (see the config-directory note above for where these live). If every server fails at almost the same millisecond with `CancelledError` (not a real error like `WinError 2` or an import traceback), this is a known transient issue — most likely a one-time delay (e.g. Windows Defender scanning a freshly-rebuilt `.exe` right after a `pip install`) tipping over a shared startup timeout across all the servers Hermes connects to at once. **Just stop it (`Ctrl+C`) and start it again** — this has resolved it every time so far. If a *specific* server keeps failing while the others connect fine, that points at a real problem with that one server instead (isolate it by temporarily commenting out just that block in `config.yaml` and confirming the others connect cleanly).

To verify a specific server is genuinely healthy independent of Hermes, connect to it directly with the real `mcp` SDK rather than guessing from Hermes's logs:
```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="algaesense-mcp-actuators", args=[], env={})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print([t.name for t in (await session.list_tools()).tools])

asyncio.run(main())
```
If this lists the tools correctly and instantly, the server itself is fine — the problem is in Hermes's own connection handling, not your code/config.

### The `Get-Command algaesense-mcp-actuators` (or any `algaesense-*`) path looks stale, or a tool that should exist isn't available

Confirm the resolved executable is actually inside this project's `.venv`, not a leftover install somewhere else on `PATH`:
```powershell
Get-Command algaesense-mcp-actuators | Select-Object -ExpandProperty Source
```
If it's missing entirely or resolves somewhere unexpected, `git pull` then reinstall (`pip install -e "packages/algaesense-agent[sftp]"`) in the same environment Hermes actually runs from.

### The dashboard link Hermes sends never opens

`localhost` in a URL always refers to whichever device opens the link, not the machine that sent it — it only works if you're opening it from the *same machine* Streamlit runs on, **and** Streamlit is actually running at that moment. `apply_start_new_experiment_run` auto-launches Streamlit if it isn't already running, but only for a `localhost`/`127.0.0.1` `ALGAESENSE_DASHBOARD_URL` — there's no way to start a process on a different machine from a URL alone, so a LAN/remote dashboard URL needs Streamlit already running there yourself (see the Startup-folder auto-start note below).

### Streamlit "randomly" stops working partway through a troubleshooting session

A Windows console-script executable (`streamlit.exe`, `algaesense-mcp-actuators.exe`, etc.) is still just a Python process underneath. Running `Get-Process python | Stop-Process -Force` to clear a stuck Hermes/MCP process **also kills Streamlit**, since `streamlit.exe` is itself a Python wrapper — not a separate, protected process. If a previously-working dashboard stops responding right after a broad process-kill, that's almost certainly why, not a new bug.

**To make Streamlit start automatically** (so you don't have to remember to relaunch it), create a `.bat` file in your Windows Startup folder:
```powershell
notepad "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\AlgaeSenseDashboard.bat"
```
with contents like:
```bat
@echo off
set ALGAESENSE_EDGE_BASE_URL=http://<pi-address>:8000
start "" "<path-to-repo>\.venv\Scripts\streamlit.exe" run "<path-to-repo>\packages\algaesense-agent\src\algaesense_agent\dashboard\streamlit_app.py"
```
This runs every time you log into Windows. To start it immediately without logging out/in, just run the `.bat` file directly: `& "$env:APPDATA\...\AlgaeSenseDashboard.bat"`.

### A Pi-side restart/sync tool fails with a permission error deleting or restarting something

If `algaesense-edge` runs as `root` on the Pi (needed for GPIO access) while you SSH in as a different, non-root user, files it writes are root-owned — deleting one needs *write* permission on its *containing directory*, not the file's own permissions. `UMask=0002` in the systemd unit is **not** sufficient (it only grants group write, and your SSH user usually isn't in the same group as root-owned files) — use `UMask=0000` instead. See `docs/remote_storage_setup.md` and `docs/remote_experiment_control.md` for the full permission setup for the pull-from-Pi sync and remote-restart features respectively.
