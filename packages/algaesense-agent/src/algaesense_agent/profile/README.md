# Hermes profile: reactor experiment assistant

This directory holds everything needed to wire a real Hermes Agent
installation up to this package's four MCP servers — none of it is
Python code Hermes imports; Hermes only ever talks to `mcp_pipeline`,
`mcp_actuators`, `dashboard`, and `labwiki` as external stdio subprocesses
over the MCP protocol.

- **`system_prompt.md`** — the actual instructions for the "reactor
  experiment assistant" persona, most importantly the propose-then-confirm
  rule for actuator changes. Point Hermes's system prompt configuration at
  this file (or paste it into a channel-specific ephemeral system prompt).
- **`hermes_config.example.yaml`** — the `mcp_servers:` block to merge
  into your own `~/.hermes/config.yaml`, plus the Slack gateway
  environment variables Hermes itself needs (these are not
  algaesense-agent settings).
- **`weekly_cron_job.md`** — the `/cron` command to register the
  labwiki-consistency check on a weekly cadence.

## Setup order

1. Install `algaesense-agent` (and its `mcp`/`jaxsr-calibration` dependencies) into the same Python environment Hermes runs in.
2. Install `hermes-agent` (`pip install hermes-agent`) if not already present — see https://hermes-agent.nousresearch.com/docs/getting-started/quickstart.
3. Merge `hermes_config.example.yaml`'s `mcp_servers:` block into `~/.hermes/config.yaml`, filling in the real paths/URLs for your installation.
4. Set the Slack environment variables documented in that same file, and enable the `message.channels`/`message.groups` event subscriptions on the Slack app.
5. Point the system prompt at `system_prompt.md`.
6. Register the weekly cron job per `weekly_cron_job.md`.
7. Message the bot in Slack to confirm the four MCP servers show up as available tools (Hermes's own `/mcp list` or equivalent command).

None of this is automated by algaesense-agent itself — v1 is "here are the pieces and the wiring instructions," not a one-command installer. See the plan's Phase 3 (packaging) for where a bootstrap CLI is intended to eventually live.
