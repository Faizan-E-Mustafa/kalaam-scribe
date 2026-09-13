# AGENTS.md

Guidance for AI-assisted coding in this repository.

## Purpose: learning

The user is an AI engineer using this project mainly to **learn new domains** (e.g.
Android development). When helping:
- Pause to explain non-obvious concepts and idioms the user may not know.
- Keep explanations **concise**; avoid walls of text.
- Where sensible, offer a **recommendation** (and note the trade-off) rather than
  only laying out options.

## AI tools

This project uses [opencode](https://opencode.ai). Custom skills live in
`.opencode/skills/` and are auto-loaded.

## Development workflow

`learning_ws` is a Python project (React optional for the frontend). It uses a
plan -> tickets -> implement -> review loop driven by skills in
`.opencode/skills/`:

1. **Plan** — turn requirements into a spec and break it into tickets:
   - `to-spec` — synthesize the current conversation into a spec.
   - `wayfinder` — map a large piece of work as decision tickets, resolved one at
     a time.
   - `grill-with-docs` — sharpens a plan/design and updates domain docs (ADR,
     glossary) as you go.
   - `triage` — move issues/PRs through a triage state machine.
2. **Tickets** — `to-tickets` splits a plan/spec into tracer-bullet tickets,
   each declaring what blocks it.
3. **Implement** — `implement` builds work from a spec/tickets and drives `tdd`
   at agreed seams; `tdd` is the red-green-refactor loop for features and bug
   fixes.
4. **Review** — `code-review` reviews the diff since a fixed point along two
   axes: coding standards and spec fidelity.
5. **Supporting** — `diagnosing-bugs`, `research`, `prototype`,
   `resolving-merge-conflicts`, `domain-modeling`, `codebase-design`,
   `improve-codebase-architecture`, `wizard`.

## Agent skills

### Issue tracker

Issues and specs for this repo live as markdown files under `.scratch/`, one
feature per directory. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles, each label string equal to its name: `needs-triage`,
`needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: one `CONTEXT.md` at the repo root plus `docs/adr/`. See
`docs/agents/domain.md`.

## Running/verifying

Follow the project's own tooling (see `README.md`, `pyproject.toml` / `Makefile`
when present) for running tests, linting, and typechecks.

## Android app

The native Android app lives in `android/`. To build the toolchain, build the APK,
connect a device, and reproduce the on-device whisper spike, follow
`android/SETUP.md` (a portable, copy-paste runbook).
