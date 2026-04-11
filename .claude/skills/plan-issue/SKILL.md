---
name: plan-issue
description: Plan the implementation of a specific GitHub issue — reads the issue, relevant docs, and enters plan mode
allowed-tools: Bash, Read, Grep, Glob, Agent, AskUserQuestion, EnterPlanMode, ExitPlanMode
---

# Plan Issue Skill

Plan the implementation of a specific GitHub issue. Reads the GitHub issue, relevant documentation, and enters plan mode for user approval before any code is written.

## Context

Current branch: !`git branch --show-current`

## Instructions

### Step 1 — Identify the issue

1. Parse `$ARGUMENTS` for a step number. It may be provided as just an issue number (e.g., `27`, `#27`), or as `Issue 27`, or with an issue link (e.g., `https://github.com/kkokosa/dotLLM/issues/27`).
2. If no issue is found in `$ARGUMENTS`, ask the user with `AskUserQuestion`.
3. If an existing issue was found, fetch its details: `gh issue view <number> --json title,body,labels,comments`
4. Use `AskUserQuestion` to confirm: "Plan implementation for Issue {issue}: {title}?"

### Step 2 — Gather context

1. Read the relevant documentation listed in the roadmap step's "Key Files" or "Description" column - it may or may not contain relevant topics
2. Check `CLAUDE.md` Documentation Index — read any docs referenced for the module being implemented.
3. Read existing source files that will be modified or extended (from issue body).

### Step 3 — Enter plan mode

Use `EnterPlanMode` to enter planning mode. Then build a comprehensive implementation plan:

#### Plan structure

```markdown
# Step {N}: {Title}

**Issue**: #{issue_number}
**Branch**: `issue/{issue_number}-{short-kebab-description}`

## Summary
<1-2 sentence overview of what this step accomplishes>

## Performance expectations (if applicable)
<If this issue is performance-related, explicitly state:>
- **What** improvement is expected (e.g., "prefill throughput", "decode latency")
- **Where** in the pipeline (e.g., "attention softmax", "FFN projections")
- **How much** (e.g., "~10-35% total inference speedup" from roadmap/paper)
- **How to measure** (e.g., "bench_compare.py before/after on SmolLM-135M and Llama-3.2-1B")
- **Baseline**: run benchmarks BEFORE implementing

## Implementation plan

### 1. Create branch
`git checkout -b issue/{issue_number}-{short-kebab-description} main`

### 2. {First logical unit of work}
- Files to create/modify: ...
- What to implement: ...
- Key design decisions: ...

### 3. {Next unit}
...

### N. Tests
- Unit tests: ...
- Integration tests (if applicable): ...

### N+1. Update roadmap & README
- Mark step as `:white_check_mark:` in `docs/ROADMAP.md` - if covers topics from there
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
