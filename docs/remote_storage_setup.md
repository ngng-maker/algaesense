# Remote storage for raw experiment data

By default, `algaesense-edge` keeps every raw Parquet file it writes on
the Pi's own SD card, and the dashboard reads past experiments from a
local copy on the operator's laptop (see the main README's "past
experiments" section). That's fine for a short experiment, but a
long-running one can accumulate a lot of hourly files, and the Pi's SD
card and a laptop's disk are both a finite, and sometimes small,
resource.

This page covers the alternative: offloading each hour's completed file
to somewhere else the moment it's written, so neither the Pi nor the
laptop needs to hold onto raw data long-term. This is **entirely
optional** — leave `--storage-backend` at its default (`none`) and
nothing here applies; every file stays local exactly as before this
feature existed.

## How it's designed to be pluggable

Every part of the codebase that touches remote storage talks to one
small interface, `jaxsr_calibration.storage.RemoteStorageBackend`
(`packages/jaxsr-calibration/src/jaxsr_calibration/storage/base.py`):

- `upload_file(local_path, remote_key)`
- `download_file(remote_key, local_path)`
- `list_keys(prefix)`

That's the whole contract. Nothing else in the codebase — not the
Pi-side writer, not the dashboard's sync CLI — knows or cares which
concrete backend is actually running underneath. Four backends exist
today, selected with `--storage-backend` (or the equivalent
`ALGAESENSE_STORAGE_BACKEND` environment variable) on both
`algaesense-edge start` and `algaesense-dashboard-sync`:

- **`none`** (the default) — no remote storage. Files stay wherever
  they're written, forever, same as always.
- **`local`** — copies files to another local directory you point it
  at. This is the "I want to use my own storage, not a cloud provider"
  option: aim it at an external drive, a NAS mount, or any other
  machine's disk mounted over the network, and it behaves identically
  to a cloud backend from every caller's point of view.
- **`firebase`** — uploads to a Firebase Storage bucket (a Google Cloud
  Storage bucket managed through a Firebase project).
- **`sftp`** — pushes files directly onto another machine over SSH (e.g.
  the Pi pushing straight onto an operator's laptop). No cloud account,
  no shared network filesystem (SMB/NFS) to set up — just an SSH server
  already running on the destination and a key pair. This is the
  simplest option if you'd rather keep data entirely between your own
  machines than involve a cloud provider.

### Adding your own backend

A future user who wants a different cloud provider (Cloudflare R2, AWS
S3, Backblaze B2, ...) doesn't need to touch the Pi/edge/dashboard code
at all — just:

1. Write a new class implementing the three methods above (see
   `local_backend.py` or `firebase_backend.py` for the pattern — each is
   under 60 lines).
2. Add one `elif` branch for it in
   `jaxsr_calibration/storage/factory.py`'s `get_storage_backend`.

Everything else (the writer's upload-then-delete logic, the dashboard
sync CLI) keeps working unchanged, since they only ever call the three
interface methods.

## What actually happens on the Pi

`PartitionedParquetWriter` (the thing that turns buffered sensor rows
into hourly Parquet files) still always writes to local disk first — a
partial hour's data has to live somewhere while it's being built up.
What changes is what happens the moment an hour's file is complete:

- **`none`**: nothing — the file stays right there, forever (today's
  behavior).
- **`local`/`firebase`**: the completed file is uploaded to the backend,
  and the *local* copy is deleted immediately afterward. If the upload
  itself fails, the local file is left in place rather than deleted with
  no remote copy to show for it, so a network hiccup never loses data.

If the Pi restarts mid-hour after an upload-and-delete already happened
for that hour, the writer notices the local file is missing, downloads
the already-uploaded partial hour back from the backend first, and
appends the new rows to it before re-uploading — the same "don't lose
data across a restart" guarantee the local-only mode already had, just
sourced from remote instead of from disk when needed.

## Setting up Firebase (the backend this project uses today)

1. Go to the [Firebase console](https://console.firebase.google.com/)
   and create a project (or use an existing one).
2. In the project, open **Build → Storage** and click **Get started** to
   provision a default Storage bucket. Note the bucket name shown (looks
   like `your-project-id.appspot.com` or `your-project-id.firebasestorage.app`
   depending on when the project was created).
3. Open **Project settings → Service accounts**, click **Generate new
   private key**. This downloads a JSON file — this is a real credential,
   treat it like a password:
   - **Never commit it to this git repo.** Keep it outside the repo
     entirely (e.g. `~/algaesense-firebase-key.json` on both the Pi and
     the laptop), or if it must live inside the repo directory for
     convenience, add its exact filename to `.gitignore` first.
   - Copy it to both the Pi (if the Pi is uploading directly) and the
     laptop (if the laptop is the one syncing/downloading), each in a
     location only that machine's user account can read.
4. Install the extra dependency on whichever machine(s) will use this
   backend: `pip install "jaxsr-calibration[cloud]"` (this pulls in
   `firebase-admin`; not needed at all for `none` or `local`).

### Configuring the Pi (`algaesense-edge start`)

Either as CLI flags or environment variables (matching this project's
existing `ALGAESENSE_*` convention):

```
algaesense-edge start \
  --experiment exp_2026-08-01_batch01 \
  --reactor R01 --sensor PID01 --camera CAM01 \
  --max-par 300.0 --par-per-full-duty 1000.0 \
  --led-gpio-pin 18 --led-num-pixels 40 --led-pixel-order BRG \
  --voc-i2c-address 0x48 \
  --storage-backend firebase \
  --storage-firebase-credentials /home/pi/algaesense-firebase-key.json \
  --storage-firebase-bucket your-project-id.appspot.com
```

Or via environment variables (handy for the systemd unit — see
`docs/hardware_setup.md`):

```
ALGAESENSE_STORAGE_BACKEND=firebase
ALGAESENSE_STORAGE_FIREBASE_CREDENTIALS=/home/pi/algaesense-firebase-key.json
ALGAESENSE_STORAGE_FIREBASE_BUCKET=your-project-id.appspot.com
```

### Pulling data down on the laptop (`algaesense-dashboard-sync`)

This replaces the manual `scp` step described in the main README's
"past experiments" section — instead of copying files off the Pi by
hand, pull them straight from Firebase:

```
algaesense-dashboard-sync \
  --data-dir ./data --db-path ./data/dashboard_history.db \
  --storage-backend firebase \
  --storage-firebase-credentials ./algaesense-firebase-key.json \
  --storage-firebase-bucket your-project-id.appspot.com
```

Add `--experiment-id exp_...` to sync just one experiment instead of
every experiment currently in the bucket. Run this any time (the
experiment doesn't need to have finished) to refresh the dashboard's
"Past experiment" view with whatever's been uploaded so far.

## Pulling from the Pi (simplest option — no new SSH server needed anywhere)

If you already SSH into the Pi (most setups do, per `docs/hardware_setup.md`), this is the least setup of any option here: the laptop pulls files from the Pi over that same, already-working connection, and deletes each file from the Pi right after it's copied. No SSH server needs to be enabled anywhere new — the Pi's own `sshd` (already running) is all this uses, and Windows already ships an SSH *client* (just not a server) out of the box.

```
algaesense-dashboard-sync --data-dir ./data --db-path ./data/dashboard_history.db \
  --pull-from-pi \
  --pi-host <pi-tailscale-address> \
  --pi-username pi \
  --pi-private-key ~/.ssh/id_ed25519 \
  --pi-remote-raw-dir /home/pi/algaesense/algaesense/data/raw
```

`--pi-private-key` is whatever key you already use to `ssh` into the Pi. If you currently log in with a **password** instead, use `--pi-password <your-password>` in place of `--pi-private-key` — this works immediately with no extra setup. Switch to a key later (`ssh-keygen` on the Pi, then append the public half to the Pi's own `~/.ssh/authorized_keys`) once you want this to run unattended/scheduled, since a scheduled task has nothing to type a password into.

Install the extra first: `pip install "algaesense-agent[sftp]"` (same `paramiko` dependency as the `sftp` push backend below, just used in the opposite direction here).

**To run this automatically** (not something you trigger by hand each time), schedule it — e.g. Windows Task Scheduler: **Create Basic Task** → trigger **Daily**, then edit the trigger to **Repeat task every** 15–60 minutes → action **Start a program**, pointing at your Python environment's `algaesense-dashboard-sync` with the arguments above. Once created, new experiment data moves from the Pi to your laptop (and clears off the Pi) on its own, with nothing left to remember to run.

## Pushing directly to your own laptop over SSH (no cloud account, no shared drive)

The option above (laptop pulls from Pi) needs the least setup for most people, since it reuses SSH access you already have. The alternative below has the Pi push instead — only worth it if you'd rather the Pi initiate the transfer the instant each hour completes, rather than on the pull schedule above.

This is the `sftp` backend: the Pi pushes each completed hour's file
straight onto your laptop the moment it's ready, over SSH, and deletes
its own local copy right after. Unlike `local`, it doesn't need a
network file share (SMB/NFS) mounted on the Pi — just an SSH server
already running on the laptop.

### 1. Enable an SSH server on the laptop (one-time, Windows)

Windows 10/11 ships an OpenSSH Server as an optional feature:

```powershell
# Run as Administrator
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
```

Confirm it's listening: `Test-NetConnection localhost -Port 22` should
report `TcpTestSucceeded : True`.

### 2. Generate a key pair on the Pi and authorize it on the laptop

```bash
# On the Pi
ssh-keygen -t ed25519 -f ~/.ssh/algaesense_sftp_key -N ""
cat ~/.ssh/algaesense_sftp_key.pub   # copy this line
```

On the laptop, append that public key line to the SSH server's
authorized-keys file for the account the Pi will log in as. For a
**non-Administrator** Windows account, that's
`C:\Users\<you>\.ssh\authorized_keys` (create the `.ssh` folder if it
doesn't exist, one public-key line per line in the file). For an
Administrator account, Windows' OpenSSH Server instead reads
`C:\ProgramData\ssh\administrators_authorized_keys` — check which
applies to your account before troubleshooting a login failure.

Test it from the Pi before wiring anything else up:
`ssh -i ~/.ssh/algaesense_sftp_key <your-windows-username>@<laptop-tailscale-address>` — should log straight in with no password prompt.

### 3. Configure the Pi

```
algaesense-edge start ... \
  --storage-backend sftp \
  --storage-sftp-host <laptop-tailscale-address> \
  --storage-sftp-username <your-windows-username> \
  --storage-sftp-private-key /home/pi/.ssh/algaesense_sftp_key \
  --storage-sftp-remote-root "C:/Users/<you>/algaesense-data/raw"
```

Install the extra on the Pi first: `pip install "jaxsr-calibration[sftp]"`.

**Important**: point `--storage-sftp-remote-root` at the `raw`
subdirectory of whatever `--data-dir` you'll later run
`algaesense-dashboard-sync` against on the laptop (e.g. if you'll run
`algaesense-dashboard-sync --data-dir C:/Users/<you>/algaesense-data`,
the remote root above should end in `.../algaesense-data/raw`). Files
then arrive already in the exact layout the dashboard's sync CLI
expects — **no `--storage-backend` flag is needed on the laptop side at
all**; just run `algaesense-dashboard-sync` normally once files have
started arriving, since they're already sitting locally by the time it runs.

## Using your own local device instead (no cloud account needed)

Point `--storage-backend local` at any directory this machine can write
to — an external drive, a mounted NAS share, or even just a different
partition:

```
algaesense-edge start ... \
  --storage-backend local \
  --storage-local-root /mnt/external-drive/algaesense-data
```

and on the laptop side:

```
algaesense-dashboard-sync --data-dir ./data --db-path ./data/dashboard_history.db \
  --storage-backend local --storage-local-root /mnt/external-drive/algaesense-data
```

No new dependency, no account, no credentials file — just another
directory.

## A note on free-tier limits (Firebase)

Firebase Storage's free (Spark) tier currently includes 5 GB of stored
data and 1 GB/day of download traffic. Raw VOC Parquet files are small
(one sensor's hourly file at ~1 Hz sampling is well under a few hundred
KB), so storage is unlikely to be the binding constraint for a typical
experiment; repeatedly re-syncing/re-downloading large campaigns for
analysis is more likely to approach the daily download cap. Check
current pricing/limits at
[firebase.google.com/pricing](https://firebase.google.com/pricing)
before relying on this for a very large or very frequently-re-analyzed
dataset.
