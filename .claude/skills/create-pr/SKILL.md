---
name: create-pr
description: Commit remaining changes, push branch, and create a PR with detailed description
disable-model-invocation: true
allowed-tools: Bash, Read, Grep, Glob, Agent
---

# Create PR Skill

Commit all remaining changes on the current branch, push to remote, and create a GitHub PR with a detailed description following dotLLM conventions.

## Context

Current branch: !`git branch --show-current`
Base branch: main
Uncommitted changes: !`git status --short`
Recent commits on this branch (not on main): !`git log main..HEAD --oneline 2>/dev/null || echo "(no commits yet)"`
Changed files vs main: !`git diff --name-only main...HEAD 2>/dev/null || echo "(no diff)"`

## Instructions

Follow these steps precisely:

### Step 1 — Commit remaining changes (if any)

1. Run `git status` to check for uncommitted changes.
2. If there ARE uncommitted changes:
   - Run `git diff` and `git diff --cached` to understand what changed.
   - Stage relevant files (do NOT stage `.claude/settings.local.json` or files containing secrets).
   - Create a commit with a message that references the issue number (extracted from branch name `issue/{N}-...`).
   - The commit message must end with `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.
3. If there are NO uncommitted changes, skip to Step 2.

### Step 2 — Push branch

1. Check if the branch has an upstream: `git rev-parse --abbrev-ref @{upstream} 2>/dev/null`
2. If no upstream, push with: `git push -u origin <branch-name>`
3. If upstream exists, push with: `git push`

### Step 3 — Create PR

1. Gather full context:
   - All commits: `git log main..HEAD --format="%h %s"`
   - Full diff summary: `git diff --stat main...HEAD`
   - Read the GitHub issue for context (extract issue number from branch name): `gh issue view <N> --json title,body,labels`
2. Draft the PR:
   - **Title**: Short (under 70 chars), descriptive of the overall change. Use the issue title as a starting point.
   - **Body**: Use this template:

```
Closes #<issue-number>

## Summary
<3-5 bullet points covering the key changes, grouped logically>

## Technical Details
<Deeper explanation of approach, design decisions, trade-offs — as much detail as warranted by the complexity>

## Key Files
| File | Changes |
|------|---------|
| `path/to/file` | Brief description |
| ... | ... |

## Test Plan
- [ ] Unit tests pass: `dotnet test tests/DotLLM.Tests.Unit/`
- [ ] Integration tests pass: `dotnet test tests/DotLLM.Tests.Integration/`
- <any additional verification steps relevant to this change>
```

3. Create the PR: `gh pr create --title "..." --body "$(cat <<'EOF' ... EOF)"`
4. Return the PR URL to the user.

### Additional arguments

If `$ARGUMENTS` is provided, incorporate it as additional context for the PR description (e.g., specific notes to highlight, labels to add, reviewers to request).
