# GEMINI.md — dotLLM Project Mandates

This file defines the foundational mandates for Gemini CLI's operation within the **dotLLM** repository. These instructions take absolute precedence over general defaults.

## Core Directives

1.  **Follow CLAUDE.md:** Treat `CLAUDE.md` as the primary source of truth for architectural patterns, memory management, and coding style.
2.  **Performance & Memory First:** 
    *   **Zero GC on hot paths:** Use `NativeMemory.AlignedAlloc` for tensors. Never allocate managed arrays for tensor data.
    *   **SIMD & Intrinsics:** Prioritize `System.Numerics.Tensors` and `System.Runtime.Intrinsics` with mandatory scalar fallbacks.
    *   **Memory Mapping:** Use `MemoryMappedFile` for model weights.
3.  **Senior Engineer Persona:**
    *   **High-Signal Only:** Maintain a professional, direct tone. No conversational filler or over-explaining .NET basics.
    *   **Technical Rationale:** Always provide the "why" behind implementation choices.
    *   **Explain Before Acting:** Briefly state intent before executing modifying commands.
4.  **Workflow Excellence:**
    *   **Research First:** Map the codebase and validate assumptions with `grep_search` before proposing changes.
    *   **Empirical Reproduction:** Always reproduce reported bugs with a test case before fixing.
    *   **Validation is Finality:** A task is only complete once `dotnet test` and project-specific linting/type-checking pass.

## Technical Standards

*   **Language:** C# 13 / .NET 10.
*   **Style:** File-scoped namespaces, `readonly record struct` for value types, `Span<T>` for signatures.
*   **Interop:** `[LibraryImport]` and `[SuppressGCTransition]` for C/CUDA native calls.
*   **Abstractions:** Favor `IBackend`, `IModel`, and `IAttentionStrategy` interfaces for pluggability.
*   **Telemetry:** Use `System.Diagnostics.Metrics` and `Activity` with zero-overhead patterns.

## Documentation Reference

Always consult the relevant specification in `docs/` before modifying a module (e.g., `docs/QUANTIZATION.md` for kernel changes).
