# labwiki conventions

This file documents how the labwiki knowledge base is organized and updated, for whoever (human or agent) is reading or writing to it directly. It mirrors this repo's own `CLAUDE.md`/memory-writing conventions, applied to per-campaign experimental knowledge instead of coding-session knowledge.

## Layout

Each campaign gets its own directory under the configured wiki root:

```
{campaign_id}/
├── raw/                  # immutable
│   └── {experiment_id}.yaml
└── wiki/
    ├── index.md           # catalog of every experiment, linked
    ├── log.md              # append-only, one line per ingest event
    ├── summaries/
    │   └── {experiment_id}.md
    ├── concepts/
    │   └── {topic}.md      # synthesized cross-experiment findings
    └── entities/
        └── {sensor_or_reactor_id}.md
```

## Raw layer

Every completed experiment's result lands in `raw/{experiment_id}.yaml` exactly once, written by `ingest_experiment_result`. This file is never edited after being written — if a correction is needed, re-run ingestion for that experiment (it overwrites its own raw file and updates the derived pages), don't hand-edit it.

## Wiki layer

- `summaries/{experiment_id}.md` — one page per experiment, generated deterministically from the raw source: conditions, results, the fit expression (if any), and operator notes. Regenerated (overwritten) each time that experiment is re-ingested.
- `entities/{id}.md` — one page per sensor or reactor. Each ingest appends (or replaces, if re-ingesting the same experiment) one line linking back to that experiment's summary. Read an entity's page to see its whole history at a glance.
- `index.md` — one line per experiment, added once per new experiment_id (not duplicated on re-ingest).
- `log.md` — append-only. Never edit past lines; this is the truthful record of *when* something was ingested, independent of what the derived pages currently say.
- `concepts/{topic}.md` — **not created automatically.** These are synthesized, cross-experiment findings ("PAR vs VOC emission — findings so far") that require judgment across multiple experiments' summaries — write these using the agent's own file-editing tools once there's enough raw material in `summaries/`/`entities/` to synthesize from, not through `ingest_experiment_result`. When creating one, cross-reference every experiment it draws on with a `[[experiment_id]]` wikilink so the lint pass (below) can confirm it's still grounded in real data.

## Cross-references

Every wiki page links to related pages using `[[double-bracket wikilinks]]` — e.g. an experiment summary links to `[[campaign_id]]`, `[[reactor_id]]`, `[[sensor_id]]`. A wikilink is just the target's identifier (file stem); there is no separate link database, so any tool reading these files can resolve one by looking for a matching filename.

## When to create a new entity page vs. append to an existing one

One entity page per distinct sensor_id/reactor_id, for the lifetime of the campaign — never create a second page for the same entity. If a sensor is rotated to a different reactor mid-campaign (see the rotation schedule in `jaxsr_calibration.calibration.config.RotationSchedule`), that's still the same entity page; the new reactor pairing shows up naturally in the linked experiment summaries.

## Lint pass

`lint_labwiki` (see `labwiki/lint.py`) runs on the same weekly cadence as `jaxsr_calibration.diagnostics.run_weekly_audit` (configured as a separate Hermes cron job, not a Python-level call between the two packages — `jaxsr_calibration` never imports from `algaesense_agent`). It checks two things mechanically:

- **Orphaned pages**: an entity or summary page that nothing in `index.md` or another page links to.
- **Stale entity pages**: an experiment was ingested naming a reactor/sensor, but that entity's page has no line referencing it.

It does **not** attempt to detect contradictory findings automatically — recognizing that two `concepts/` pages disagree requires reading and understanding them, which is left to the agent's own judgment when it reviews whatever the mechanical checks flag.
