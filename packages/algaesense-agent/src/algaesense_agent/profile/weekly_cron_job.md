# Weekly labwiki consistency cron job

Registered directly through Hermes's own `/cron` command (there is no
algaesense-agent code involved in scheduling — Hermes's gateway daemon
owns the scheduler, per its cron feature).

In a Hermes chat session, once the `algaesense_labwiki` MCP server (see
`hermes_config.example.yaml`) is registered:

```
/cron add --schedule "0 9 * * 1" --prompt "For each active AlgaeSense campaign, call lint_labwiki_consistency with that campaign_id via the algaesense_labwiki tools. Post a summary to this channel: for each campaign, either 'no issues found' or the exact list of warnings returned. Do not attempt to fix anything yourself -- just report."
```

`"0 9 * * 1"` is a standard cron expression for every Monday at 9am — adjust for your own campaign's cadence.

## Why the prompt lists the campaign_ids explicitly (or is written to ask)

Cron jobs run in a completely fresh agent session with no memory of prior chat — the prompt above must either name the campaign_ids directly (edit it in whenever a new campaign starts) or instruct the agent to first discover them (e.g. by listing `data/derived/features/*/` directories, if you also wire a small filesystem-listing tool for that). v1 keeps this manual — edit the `/cron` job's prompt text yourself when a campaign starts or ends, the same way you'd update any other standing instruction.

## Not yet wired: `jaxsr_calibration.diagnostics.run_weekly_audit`

The plan's original Phase 2 design intended this same cadence to also trigger `run_weekly_audit`'s sensor-health checks (fleet-zero, ambient baseline, swap-pilot summary). That function still requires a `--input <parquet file>` of already-collected diagnostic data and isn't wrapped as an MCP tool yet — for now, run `jaxsr-cal diagnose weekly-audit` yourself and treat the cron job above as covering the labwiki-consistency half only. Wrapping the diagnostics themselves as a tool (so this cron job can cover both) is a natural next addition to `mcp_pipeline` or a new `mcp_diagnostics` server, not yet built.
