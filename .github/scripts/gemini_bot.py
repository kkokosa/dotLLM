#!/usr/bin/env python3
"""
Gemini bot for dotLLM GitHub reviews.
Triggered by @gemini mentions in PR/issue comments, review comments, and issues.

Model is controlled by the GEMINI_MODEL repository variable (Settings → Variables).
Set it to any model ID (e.g. gemini-2.5-pro, gemini-2.0-flash-exp) without
touching code. Defaults to gemini-2.5-pro if the variable is unset.
"""

import json
import os
import sys
import subprocess
import tempfile
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_DIFF_CHARS = 60_000
MAX_COMMENT_CHARS = 2_000
MAX_THREAD_COMMENTS = 15

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a code reviewer and research assistant for dotLLM, an open-source \
high-performance LLM inference engine written in C#/.NET. It targets \
transformer-based models (Llama, Mistral, Phi, Qwen, DeepSeek) with CPU \
(SIMD-optimized) and CUDA GPU backends. The codebase prioritises unmanaged \
memory, zero GC pressure on the inference hot path, and SIMD vectorisation \
via System.Runtime.Intrinsics and System.Numerics.Tensors.

You are responding to a GitHub comment that tagged @gemini. \
Be concise and technically precise. \
Format your response as GitHub-flavored Markdown. \
Focus on correctness, performance, and .NET best practices. \
Reference specific file names and line numbers when relevant.\
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sh(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def extract_prompt(body: str) -> str:
    """Return the text that follows the first @gemini mention."""
    lower = body.lower()
    idx = lower.find('@gemini')
    if idx < 0:
        return body.strip()
    after = body[idx + len('@gemini'):].strip()
    return after if after else body.strip()


def get_pr_context(repo: str, pr_number: int) -> tuple[dict, str]:
    """Return (pr_metadata_dict, diff_string). Either may be empty on failure."""
    pr_result = sh(['gh', 'pr', 'view', str(pr_number),
                    '--json', 'title,body,headRefName,baseRefName'])
    pr = json.loads(pr_result.stdout) if pr_result.returncode == 0 else {}

    diff_result = sh(['gh', 'pr', 'diff', str(pr_number)])
    diff = diff_result.stdout if diff_result.returncode == 0 else ''
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + f'\n\n[diff truncated — first {MAX_DIFF_CHARS} chars shown]'

    return pr, diff


def get_pr_thread(repo: str, pr_number: int) -> str:
    """Return recent PR conversation comments as a formatted string."""
    result = sh(['gh', 'api', f'repos/{repo}/issues/{pr_number}/comments',
                 '--jq', '[.[] | {login: .user.login, body: .body}]'])
    if result.returncode != 0:
        return ''
    try:
        comments = json.loads(result.stdout)
        parts = []
        for c in comments[-MAX_THREAD_COMMENTS:]:
            body = c['body'][:MAX_COMMENT_CHARS]
            parts.append(f"**@{c['login']}**: {body}")
        return '\n\n'.join(parts)
    except Exception:
        return ''


def call_gemini(api_key: str, prompt: str) -> str:
    """Call Gemini and return the text response. Model read from GEMINI_MODEL env var."""
    model = os.environ.get('GEMINI_MODEL', '').strip() or 'gemini-2.5-pro'
    print(f'Using model: {model}')
    url = ('https://generativelanguage.googleapis.com/v1beta/'
           f'models/{model}:generateContent?key={api_key}')
    gen_config: dict = {
        'maxOutputTokens': 65535,
        'temperature': 0.2,
    }
    # Thinking models (2.5+, 3.x) use thinking tokens that eat into maxOutputTokens.
    # Cap thinking budget so the actual response isn't truncated.
    # thinkingBudget works on both 2.5 and 3.x series.
    if any(tag in model for tag in ('2.5', '3.0', '3.1', '3.5')):
        gen_config['thinkingConfig'] = {'thinkingBudget': 4096}

    payload = {
        'system_instruction': {'parts': [{'text': SYSTEM_PROMPT}]},
        'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
        'generationConfig': gen_config,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read())
            candidate = body['candidates'][0]
            finish_reason = candidate.get('finishReason', 'UNKNOWN')
            if finish_reason not in ('STOP', 'UNKNOWN'):
                print(f'WARNING: Gemini finish_reason={finish_reason} — response may be truncated')
            # Thinking models return multiple parts; extract the last text part
            # (thinking parts have no 'text' key or have a 'thought' flag).
            parts = candidate['content']['parts']
            for part in reversed(parts):
                if 'text' in part and not part.get('thought'):
                    return part['text']
            return parts[-1].get('text', '')
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            if e.code == 503 and attempt < 2:
                wait = 15 * (attempt + 1)  # 15s, then 30s
                print(f'Gemini 503 (attempt {attempt + 1}/3) — retrying in {wait}s...')
                time.sleep(wait)
                continue
            raise RuntimeError(f'Gemini API error {e.code}: {error_body}') from e
        except OSError as e:
            # Catches TimeoutError, ConnectionResetError, RemoteDisconnected, etc.
            # — all network-layer failures inherit from OSError.
            if attempt < 2:
                wait = 20 * (attempt + 1)  # 20s, then 40s
                print(f'Gemini network error ({type(e).__name__}, attempt {attempt + 1}/3)'
                      f' — retrying in {wait}s...')
                time.sleep(wait)
                continue
            raise RuntimeError(f'Gemini API unreachable after 3 attempts: {e}') from e


def post_reply(event_name: str, event: dict, repo: str, body: str):
    """Post body as a comment/reply appropriate for the triggering event."""
    # Write body to a temp file to avoid shell command-line length limits
    # and to safely handle Markdown with quotes, newlines, etc.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(body)
        tmpfile = f.name

    try:
        if event_name == 'issue_comment':
            number = event['issue']['number']
            is_pr = 'pull_request' in event['issue']
            if is_pr:
                sh(['gh', 'pr', 'comment', str(number), '--body-file', tmpfile], check=True)
            else:
                sh(['gh', 'issue', 'comment', str(number), '--body-file', tmpfile], check=True)

        elif event_name == 'pull_request_review_comment':
            # Reply inline to the specific review comment.
            comment_id = event['comment']['id']
            pr_number = event['pull_request']['number']
            with open(tmpfile) as fh:
                body_text = fh.read()
            sh(['gh', 'api',
                f'repos/{repo}/pulls/{pr_number}/comments',
                '-X', 'POST',
                '-f', f'body={body_text}',
                '-F', f'in_reply_to={comment_id}'], check=True)

        elif event_name == 'pull_request_review':
            pr_number = event['pull_request']['number']
            sh(['gh', 'pr', 'comment', str(pr_number), '--body-file', tmpfile], check=True)

        elif event_name == 'issues':
            issue_number = event['issue']['number']
            sh(['gh', 'issue', 'comment', str(issue_number), '--body-file', tmpfile], check=True)
    finally:
        os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        print('GEMINI_API_KEY is not set.', file=sys.stderr)
        sys.exit(1)

    event_name = os.environ['GITHUB_EVENT_NAME']
    repo = os.environ['GITHUB_REPOSITORY']

    with open(os.environ['GITHUB_EVENT_PATH']) as f:
        event = json.load(f)

    # -----------------------------------------------------------------------
    # Extract triggering text, commenter, and PR/issue number
    # -----------------------------------------------------------------------
    if event_name == 'issue_comment':
        raw_body = event['comment']['body']
        commenter = event['comment']['user']['login']
        is_pr = 'pull_request' in event['issue']
        number = event['issue']['number']

    elif event_name == 'pull_request_review_comment':
        raw_body = event['comment']['body']
        commenter = event['comment']['user']['login']
        is_pr = True
        number = event['pull_request']['number']

    elif event_name == 'pull_request_review':
        raw_body = event['review']['body'] or ''
        commenter = event['review']['user']['login']
        is_pr = True
        number = event['pull_request']['number']

    elif event_name == 'issues':
        raw_body = (event['issue']['title'] + '\n\n' +
                    (event['issue']['body'] or ''))
        commenter = event['issue']['user']['login']
        is_pr = False
        number = event['issue']['number']

    else:
        print(f'Unsupported event type: {event_name}', file=sys.stderr)
        sys.exit(1)

    # Guard: never respond to ourselves
    if commenter in ('github-actions[bot]', 'gemini-bot[bot]'):
        print('Skipping bot-generated comment.')
        return

    prompt_text = extract_prompt(raw_body)
    if not prompt_text:
        print('Empty prompt after @gemini — nothing to do.')
        return

    # -----------------------------------------------------------------------
    # Build context for Gemini
    # -----------------------------------------------------------------------
    sections: list[str] = [f'**Question from @{commenter}:**\n\n{prompt_text}']

    if is_pr:
        pr, diff = get_pr_context(repo, number)
        if pr:
            pr_header = f'**PR #{number} — {pr.get("title", "")}**'
            pr_body = (pr.get('body') or '')[:3_000]
            sections.append(f'{pr_header}\n\n{pr_body}')

        thread = get_pr_thread(repo, number)
        if thread:
            sections.append(f'**Recent PR discussion (newest last):**\n\n{thread}')

        if diff:
            sections.append(f'**PR diff:**\n\n```diff\n{diff}\n```')
    else:
        issue_title = event['issue']['title']
        issue_body = (event['issue']['body'] or '')[:3_000]
        sections.append(f'**Issue #{number} — {issue_title}**\n\n{issue_body}')

    full_prompt = '\n\n---\n\n'.join(sections)
    print(f'Prompt length: {len(full_prompt)} chars. Calling Gemini...')

    # -----------------------------------------------------------------------
    # Call Gemini and post reply
    # -----------------------------------------------------------------------
    try:
        response = call_gemini(api_key, full_prompt)
        reply = f'**Gemini** ✦\n\n{response}'
    except RuntimeError as e:
        msg = str(e)
        if '503' in msg:
            reply = (
                '**Gemini** ✦\n\n'
                '> ⚠️ The Gemini API is temporarily unavailable (503 — high demand). '
                'Please retry your `@gemini` comment in a few minutes.'
            )
        elif 'unreachable' in msg or 'timed out' in msg:
            reply = (
                '**Gemini** ✦\n\n'
                '> ⏱️ The Gemini API did not respond after 3 attempts (timeout, '
                'disconnection, or overload). Please retry your `@gemini` comment '
                'in a few minutes.'
            )
        else:
            reply = f'**Gemini** ✦\n\n> ❌ Unexpected API error: `{msg}`'
        print(f'API error, posting notice: {msg}', file=sys.stderr)

    post_reply(event_name, event, repo, reply)
    print('Reply posted successfully.')


if __name__ == '__main__':
    main()
