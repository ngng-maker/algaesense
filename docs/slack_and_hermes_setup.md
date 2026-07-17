# Setting up Hermes Agent, your Anthropic key, and the Slack app

This is the walkthrough for the two things only you can do (I have no way to create a Slack app or hold your API key myself): installing Hermes, wiring in your Anthropic key, and creating the Slack app/bot token. Do this on whichever machine will actually run Hermes (the "brain" server per the architecture — could be this laptop, could be somewhere else; it does not need to be the Raspberry Pi).

**A note on secrets:** don't paste your Anthropic API key or Slack tokens into a chat with me. Every step below has you run a command yourself in your own terminal, where the secret goes straight into Hermes's own config file (`~/.hermes/.env`) and never passes through this conversation.

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

## 3. Register this project's four MCP servers

Install `algaesense-agent` into the same Python environment Hermes runs in:

```
pip install -e packages/algaesense-agent
```

Then merge the `mcp_servers:` block from [`packages/algaesense-agent/src/algaesense_agent/profile/hermes_config.example.yaml`](../packages/algaesense-agent/src/algaesense_agent/profile/hermes_config.example.yaml) into your own `~/.hermes/config.yaml`, filling in the real `ALGAESENSE_DATA_DIR`/`ALGAESENSE_EDGE_BASE_URL`/`ALGAESENSE_LABWIKI_ROOT` paths for your setup.

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
