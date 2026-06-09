# Task Router

Use this file to classify short or ambiguous user prompts before exploring the repository.

## Default Protocol

1. Classify the request into one task type below.
2. Read only the route-required files first.
3. Search or read source files only after locating relevant entrypoints.
4. Do not scan `reports/**`, `logs/**`, or `outputs/**` broadly.
5. If the route does not fit, ask one clarifying question or use the closest safe route.

## Routes

### casual_or_unclear

Examples: `hello`, `help`, ambiguous one-word prompts, or prompts with conflicting intents.

Read first:

- `docs/context/PROJECT_STATE.md`
- `docs/context/CURRENT_WORK.md`
- `docs/context/TASK_ROUTER.md`

Then:

- Ask one clarifying question if intent remains unclear.
- Do not assume implementation by default.

Avoid:

- broad repo scans
- full report scans
- raw logs unless a named failure is provided

### status_question

Examples: `what did we do`, `where are we`, `current status`.

Read first:

- `docs/context/PROJECT_STATE.md`
- `docs/context/CURRENT_WORK.md`
- `docs/context/REPORT_INDEX.md`

Avoid:

- raw logs
- full report scans
- source code reads unless the user asks for implementation details

### continue_work

Examples: `continue`, `next`, `proceed`.

Read first:

- `docs/context/PROJECT_STATE.md`
- `docs/context/CURRENT_WORK.md`
- `docs/context/TASK_ROUTER.md`

Then:

- If `CURRENT_WORK.md` has a clear active scope, continue that scope.
- If stale or ambiguous, ask one clarifying question.

### implementation

Examples: `implement this`, `fix this feature`, `add support for X`.

Read first:

- `docs/context/PROJECT_STATE.md`
- `docs/context/CURRENT_WORK.md`
- `docs/context/KNOWN_FAILURES.md` when touching high-risk areas
- relevant architecture docs only when needed

Then search:

- exact symbols, file names, or domain terms related to the task
- nearby tests for the touched code

Avoid:

- benchmark outputs unless the task is benchmark-specific
- raw logs unless debugging a named failure

Ask if:

- public API/schema may change
- production defaults may change
- backward compatibility is unclear
- multiple reasonable designs exist and no constraint selects one

### bug_report

Examples: pasted traceback, failing command, wrong answer case.

Read first:

- `docs/context/KNOWN_FAILURES.md`
- `docs/context/PROJECT_STATE.md`
- relevant error report only if `REPORT_INDEX.md` points to one

Then search:

- exact error text
- failing test name
- named example ID
- relevant source entrypoint

Avoid:

- broad raw log reads unless the failure names a log or example ID

### code_review

Examples: `review this`, `check my branch`, `find risks`.

Read first:

- changed files
- nearby tests
- relevant ADRs if authority boundaries or architecture are touched

Focus on:

- bugs
- regressions
- missing tests
- safety boundary violations

Avoid:

- unrelated reports
- broad project exploration

### benchmark_eval

Examples: `run benchmark`, `score predictions`, `compare models`.

Read first:

- `docs/context/REPORT_INDEX.md`
- relevant benchmark config
- latest benchmark/eval summary only

Avoid:

- changing prompts, datasets, or scoring during a benchmark unless explicitly requested
- raw prediction files unless scoring/debugging named failures

### docs_update

Examples: `update docs`, `write context`, `summarize decisions`.

Read first:

- target docs
- `docs/context/PROJECT_STATE.md`
- `docs/context/REPORT_INDEX.md` when summarizing reports

Then:

- keep docs concise
- link to detailed reports instead of copying them
- update indexes if new important docs are added

### architecture_decision

Examples: `should we change architecture`, `decide policy`, `make this default`.

Read first:

- `docs/context/DECISION_INDEX.md`
- relevant ADRs
- `docs/context/PROJECT_STATE.md`
- `docs/context/KNOWN_FAILURES.md`

Ask if:

- the decision changes production defaults
- constraints conflict
- evidence is incomplete

### handoff

Examples: `summarize for next session`, `handoff`, `save context`.

Read first:

- `docs/runbooks/session-handoff.md`
- `docs/context/CURRENT_WORK.md`
- `docs/context/PROJECT_STATE.md`

Then update:

- `PROJECT_STATE.md` if status changed
- `CURRENT_WORK.md` if scope changed
- relevant indexes

### self_debug

Examples: `self-debug`, `find remaining bugs`, `run debug loop`, `critique your work`.

Read first:

- `docs/runbooks/self-debug-loop.md`
- `docs/evolution/RELEASE_GATES.md`
- `docs/context/CONTEXT_DEBT.md`
- relevant tests for the changed area

Then:

- Run existing tests.
- Add or run adversarial cases.
- Perform critique before declaring done.

Avoid:

- production default changes without explicit approval
- frozen dataset or scorer edits

### project_evolution

Examples: `evolve the project`, `run next improvement loop`, `harden the project`, `make the project better`.

Read first:

- `docs/evolution/PROJECT_EVOLUTION.md`
- `docs/evolution/EVOLUTION_BACKLOG.md`
- `docs/evolution/HYPOTHESIS_LEDGER.md`
- `docs/evolution/SCORECARD.md`
- `docs/evolution/RISK_REGISTER.md`
- `docs/runbooks/project-evolution-loop.md`

Then:

- Select one bounded hypothesis.
- Define success metric and forbidden scope.
- Run self-debug and evolution gates before closeout.

Avoid:

- multi-scope autonomous edits
- production default changes without explicit approval
- benchmark protocol changes mixed with implementation changes

## Expansion Rule

If the initial route is insufficient, expand context in this order:

1. Read an index file.
2. Read one named report or architecture doc.
3. Search exact symbols or example IDs.
4. Read raw logs or JSONL only for a named failure.

Do not skip directly from a short prompt to broad scans.
