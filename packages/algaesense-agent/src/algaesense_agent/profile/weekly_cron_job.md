# Weekly diagnostics + labwiki consistency cron job

Registered directly through Hermes's own `/cron` command (there is no
algaesense-agent code involved in scheduling — Hermes's gateway daemon
owns the scheduler, per its cron feature).

In a Hermes chat session, once the `algaesense_labwiki` and
`algaesense_diagnostics` MCP servers (see `hermes_config.example.yaml`)
are registered:

```
/cron add --schedule "0 9 * * 1" --prompt "For each active AlgaeSense campaign/experiment: (1) call lint_labwiki_consistency with that campaign_id via the algaesense_labwiki tools, and (2) call run_fleet_zero_check and run_ambient_baseline_check with the relevant clean-air/ambient-air experiment_id via the algaesense_diagnostics tools, if such a diagnostic run exists for that week. Post one summary to this channel covering both: labwiki issues (or 'no issues found'), and sensor-health status per sensor (GREEN/YELLOW/RED, or 'no diagnostic run available this week'). Do not attempt to fix anything yourself -- just report."
```

`"0 9 * * 1"` is a standard cron expression for every Monday at 9am — adjust for your own campaign's cadence.

## Why the prompt lists campaign/experiment IDs explicitly (or is written to ask)

Cron jobs run in a completely fresh agent session with no memory of prior chat — the prompt above must either name the relevant IDs directly (edit it in whenever a new campaign starts, or a new clean-air/ambient diagnostic run is logged) or instruct the agent to first discover them (e.g. by listing `data/derived/features/*/` or `data/raw/experiments/*/` directories, if you also wire a small filesystem-listing tool for that). v1 keeps this manual — edit the `/cron` job's prompt text yourself as campaigns and diagnostic runs come and go, the same way you'd update any other standing instruction.

## `run_swap_pilot_check` and `run_weekly_audit_check` are not part of the automatic cron prompt above

Both exist as real tools (`algaesense_diagnostics`), but neither fits a fully-automatic weekly trigger the same way fleet-zero/ambient do:

- `run_swap_pilot_check` needs a specific sensor/reactor rotation experiment to have been run that week — not something to assume happened on a fixed schedule.
- `run_weekly_audit_check` composes *previous* swap-pilot results (`swap_pilot_variance_shares`, oldest first) plus a `sensors.yaml` path — the agent (or you, in chat) needs to supply the swap-pilot history explicitly, since there's no automatic "remember last week's swap-pilot result" store yet.

For now, run these two conversationally when there's an actual rotation experiment to review, rather than adding them to the unattended cron prompt.
