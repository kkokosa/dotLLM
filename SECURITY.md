# Security Policy

## Supported versions

dotLLM is in active development. Security fixes are applied to the latest released version only.

| Version | Supported |
|---------|-----------|
| Latest release | Yes |
| Older releases | No  |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report privately via GitHub Security Advisories:

1. Go to https://github.com/kkokosa/dotLLM/security/advisories/new
2. Fill in the report. Include a clear description, reproduction steps, affected version(s), and any suggested mitigation.
3. You will receive an acknowledgement within a few days.

If you cannot use GitHub Security Advisories, contact the maintainer via the email listed on https://kokosa.dev/.

## Scope

In scope:

- The dotLLM libraries (`DotLLM.*` NuGet packages) and the `dotllm` CLI.
- The OpenAI-compatible API server (`DotLLM.Server`, `dotllm serve`).
- The CUDA PTX kernels under `native/kernels/`.
- The release artifacts published on the [Releases page](https://github.com/kkokosa/dotLLM/releases) and https://www.nuget.org/.

Out of scope:

- Third-party models you run through dotLLM (prompt-injection, jailbreaks, misuse — these are model-level concerns, not dotLLM vulnerabilities).
- The companion website at https://dotllm.dev/ — report issues against that property directly.
- Vulnerabilities requiring local code execution or physical access.

## Disclosure

After a fix is released we will publish a GitHub Security Advisory crediting the reporter (unless anonymity is requested).
