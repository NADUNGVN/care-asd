# User–Codex–Server collaboration protocol

This document defines how CARE-ASD work is coordinated between the researcher,
Codex, and a compute server (for example `SERVER-02`). Its purpose is to keep
experiments reproducible while allowing Codex to inspect the evidence produced
on a server it cannot directly access.

## Responsibilities

| Party | Responsibility |
|---|---|
| Researcher | Runs the supplied command on the selected server and reports a failure immediately. Reviews any action that would access evaluation data or external services. |
| Codex | Makes code/docs changes, commits and pushes reviewed source changes, supplies server commands, and verifies committed server artifacts after pulling them. |
| Server | Provides isolated compute/storage. It must not be treated as the source of truth for code: all source changes and reviewable reports return to Git. |

## Command contract

- Every command supplied for the researcher to paste into a shell is **one physical line**.
- A task may contain several one-line commands, but no command relies on a variable created by a previous command unless that variable is set again on the same line.
- Commands use `&&` so a failed prerequisite prevents later actions.
- Commands that may take material time must write a named report under `reports/server/` and end by committing and pushing that report. Codex then pulls the commit and validates it.
- Each report filename contains the server identifier and a UTC timestamp or run ID. Reports must state the git SHA, command, exit status, environment summary, and paths/hashes of relevant inputs.
- Never commit raw audio, dataset archives, model checkpoints, tokens, private paths, or evaluation labels. `data/raw/`, `outputs/`, logs, and secrets stay on the server.

## Standard server lifecycle

1. Codex pushes source changes to `main`.
2. The researcher runs a one-line sync/setup command on the server.
3. The researcher runs the requested one-line task command.
4. The command writes a small, reviewable artifact in `reports/server/`, commits it, and pushes it to `main`.
5. Codex pulls the artifact, checks it against the intended contract, then issues the next task.

The server must use a clean worktree before a new task. If it is not clean, the
researcher should stop and send the output rather than discarding changes.

## Required preflight artifact

Run this after cloning/syncing a server, and after a material environment
change. It produces only non-secret metadata for Codex to inspect.

```bash
cd ~/Dung_TDTU/CARE_ASD && git pull --ff-only && mkdir -p reports/server && RUN_ID="server02_preflight_$(date -u +%Y%m%dT%H%M%SZ)" && { printf '# CARE-ASD server preflight\n\n'; printf 'git_sha='; git rev-parse HEAD; printf 'git_status=\n'; git status --short; printf '\npython=\n'; python3 --version; printf '\nuv=\n'; uv --version; printf '\ndisk=\n'; df -h .; printf '\ngpu=\n'; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; printf '\ncare_asd_environment=\n'; uv run care-asd env-report; } > "reports/server/${RUN_ID}.md" && git add "reports/server/${RUN_ID}.md" && git commit -m "report: add ${RUN_ID}" && git push origin main
```

If `uv` is not installed, report that failure; Codex will provide a separate
one-line installation command. Do not silently substitute a different Python
environment.

## Task report pattern

For any future server task, Codex supplies a command following this pattern:

```bash
cd ~/Dung_TDTU/CARE_ASD && git pull --ff-only && RUN_ID="server02_<task>_$(date -u +%Y%m%dT%H%M%SZ)" && mkdir -p reports/server && (<task command>; TASK_STATUS=$?; printf '\nexit_status=%s\n' "$TASK_STATUS"; exit "$TASK_STATUS") > "reports/server/${RUN_ID}.log" 2>&1; TASK_STATUS=$?; git add "reports/server/${RUN_ID}.log" && git commit -m "report: add ${RUN_ID}" && git push origin main; exit "$TASK_STATUS"
```

Replace `<task>` and `<task command>` only with values supplied by Codex. For a
failing task, this pattern still commits its short error output for inspection,
then returns the original failure status. Do **not** commit raw data or a huge
log.

## Dataset-specific rules

- Download the development split first; the normal path is download, extract,
  manifest, then validate.
- Retain checksum sidecars and Parquet manifests. They are small provenance
  artifacts and may be committed when reviewed.
- Evaluation data requires the documented freeze decision and explicit policy
  acceptance. It is never used for tuning and its raw content is never pushed.
- Before a large download, include free-space output in the server report.

## Completion rule

Codex considers a server task verified only after the matching report commit is
available on the remote and has been inspected. A terminal transcript pasted in
chat is useful for triage but is not the durable experiment record.
