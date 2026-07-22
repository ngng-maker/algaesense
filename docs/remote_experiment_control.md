# Starting a new experiment run from Slack

`propose_start_new_experiment_run`/`apply_start_new_experiment_run` (in `mcp_actuators`) let you ask the Slack assistant to start a fresh experiment run, rather than SSHing into the Pi and restarting `algaesense-edge` by hand. Since this **stops whatever experiment is currently running** before starting a new one, it follows the same propose → confirm → apply pattern as an LED change — the assistant will describe what it's about to do and wait for your explicit yes.

Under the hood: it SSHes into the Pi (the same connection `algaesense-dashboard-sync --pull-from-pi` already uses) and runs `sudo systemctl restart algaesense-edge`. The systemd unit already generates a fresh experiment_id from the current date/time on every restart (see `docs/hardware_setup.md`), so nothing else needs to change for a "new" experiment to actually begin.

## Configuration (on the machine running `algaesense-mcp-actuators`, i.e. wherever Hermes runs)

Reuses the exact same `ALGAESENSE_PI_*` environment variables as the dashboard's `--pull-from-pi` sync (see `docs/remote_storage_setup.md`) — if you've already set those up, there's nothing new to configure for the SSH connection itself:

- `ALGAESENSE_PI_HOST`
- `ALGAESENSE_PI_USERNAME`
- `ALGAESENSE_PI_PRIVATE_KEY` (or `ALGAESENSE_PI_PASSWORD`)
- `ALGAESENSE_PI_PORT` (defaults to 22)

One new one, so the assistant can hand you a clickable link once the restart succeeds:

- `ALGAESENSE_DASHBOARD_URL` — e.g. `http://localhost:8501` if Slack and the dashboard run on the same machine, or the machine's LAN address if you'll open Slack from somewhere else on the same network.

## One-time Pi-side setup: passwordless sudo for exactly this command

Running `sudo systemctl restart algaesense-edge` over a scripted SSH connection needs to happen without a password prompt — there's no terminal on the other end to type one into. This needs a **narrowly-scoped** sudoers rule, not blanket passwordless sudo:

```bash
sudo visudo -f /etc/sudoers.d/algaesense-restart
```

Add this single line (replace `ytpio` with whatever user you SSH in as):

```
ytpio ALL=(root) NOPASSWD: /usr/bin/systemctl restart algaesense-edge
```

Save and exit. This grants passwordless sudo for **only** that exact command — not a general `NOPASSWD: ALL`, which would let anything logging in as that user run arbitrary commands as root. Confirm the exact path to `systemctl` matches your system first: `which systemctl` (usually `/usr/bin/systemctl` or `/bin/systemctl` on Raspberry Pi OS).

Test it manually before trying it from Slack:

```bash
sudo systemctl restart algaesense-edge
```

If that runs without prompting for a password (when you're already logged in as the SSH user, not root), the sudoers rule is working.

## Safety notes

- This restarts the whole acquisition process — any in-progress calibration session, control profile, or partially-buffered hour of data gets interrupted (buffered rows not yet flushed to disk are lost, the same as any other unplanned restart; see `docs/hardware_setup.md`'s discussion of the writer's restart-recovery behavior for what *is* preserved).
- The assistant will always propose this and wait for confirmation first — if it doesn't, treat that as a bug, not something to work around.
