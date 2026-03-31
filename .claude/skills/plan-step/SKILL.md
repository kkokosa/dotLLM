---
name: plan-step
description: Plan the implementation of a roadmap step — reads the issue, relevant docs, and enters plan mode
allowed-tools: Bash, Read, Grep, Glob, Agent, AskUserQuestion, EnterPlanMode, ExitPlanMode
---

# Plan Step Skill

Plan the implementation of a roadmap step from `docs/ROADMAP.md`. Reads the GitHub issue, relevant documentation, and enters plan mode for user approval before any code is written.

## Context

Current branch: !`git branch --show-current`
Roadmap steps: !`grep -E '^\| [0-9]+ \|' docs/ROADMAP.md | head -40`

## Instructions

### Step 1 — Identify the step and issue

1. Parse `$ARGUMENTS` for a step number. It may be provided as just a number (e.g., `27`), or as `Step 27`, or with an issue number (e.g., `27 #64`).
2. If no step number is found in `$ARGUMENTS`, ask the user with `AskUserQuestion`.
3. Look up the step in `docs/ROADMAP.md` to get its title, description, and dependencies.
4. If the step already has `:white_check_mark:`, warn the user that this step appears completed.
5. Find the corresponding GitHub issue:
   - If an issue number was provided in `$ARGUMENTS`, use that.
   - Otherwise, search: `gh issue list --search "Step {N}" --json number,title,state,labels --limit 10`
   - Match by step number in the title (e.g., "Step 27:" or "Step 27 —").
   - If no issue is found, note that the issue needs to be created — the plan will include creating it as the first step (per CLAUDE.md: "every task starts with a GitHub issue").
   - If multiple matches, pick the open one. If ambiguous, ask the user.
6. If an existing issue was found, fetch its details: `gh issue view <number> --json title,body,labels,comments`
7. Use `AskUserQuestion` to confirm: "Plan implementation for Step {N}: {title} (Issue #{issue} / issue will be created)?"

### Step 2 — Gather context

1. Read the relevant documentation listed in the roadmap step's "Key Files" or "Description" column.
2. Check `CLAUDE.md` Documentation Index — read any docs referenced for the module being implemented.
3. Read existing source files that will be modified or extended (from roadmap's Key Files and issue body).
4. Check dependency steps — verify they are completed (`:white_check_mark:` in roadmap).
5. If the step depends on incomplete steps, warn the user.

### Step 3 — Enter plan mode

Use `EnterPlanMode` to enter planning mode. Then build a comprehensive implementation plan:

#### Plan structure

```markdown
# Step {N}: {Title}

**Issue**: #{issue_number}
**Branch**: `issue/{issue_number}-{short-kebab-description}`

## Summary
<1-2 sentence overview of what this step accomplishes>

## Dependencies
- Step X: {title} — {status}
- ...

## Performance expectations (if applicable)
<If this step is performance-related, explicitly state:>
- **What** improvement is expected (e.g., "prefill throughput", "decode latency")
- **Where** in the pipeline (e.g., "attention softmax", "FFN projections")
- **How much** (e.g., "~10-35% total inference speedup" from roadmap/paper)
- **How to measure** (e.g., "bench_compare.py before/after on SmolLM-135M and Llama-3.2-1B")
- **Baseline**: run benchmarks BEFORE implementing

## Implementation plan

### 1. Create GitHub issue (if none exists)
Draft the issue title, body (with acceptance criteria derived from the roadmap description and docs), and create it via `gh issue create`. Use the returned issue number for the branch name.

### 2. Create branch
`git checkout -b issue/{issue_number}-{short-kebab-description} main`

### 3. {First logical unit of work}
- Files to create/modify: ...
- What to implement: ...
- Key design decisions: ...

### 4. {Next unit}
...

### N. Tests
- Unit tests: ...
- Integration tests (if applicable): ...

### N+1. Update roadmap & README
- Mark step as `:white_check_mark:` in `docs/ROADMAP.md`
- Update `README.md` roadmap table step count
- Add News entry if significant milestone

## Key design decisions
- Decision 1: {choice} — {rationale}
- ...

## Open questions
- Any uncertainties to resolve during implementation
```

#### Plan guidelines

- Follow CLAUDE.md conventions (file-scoped namespaces, `readonly record struct`, etc.)
- Reference specific line numbers in existing files when extending them.
- For SIMD work: plan scalar reference first, then SIMD optimization.
- For new kernel work: plan correctness tests against scalar reference.
- Keep the plan concrete — specify method signatures, file paths, data structures.
- If the issue body has acceptance criteria, map each criterion to a plan section.

### Step 4 — Present plan

Use `ExitPlanMode` to present the plan for user approval. The user will review and either approve or request changes.
