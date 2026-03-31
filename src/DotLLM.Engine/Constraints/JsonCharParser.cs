using System.Runtime.CompilerServices;

namespace DotLLM.Engine.Constraints;

/// <summary>
/// Stack-based character-level JSON validator (pushdown automaton).
/// Tracks parser state and nesting context to determine which characters are valid
/// at each position in a JSON document per RFC 8259.
/// </summary>
/// <remarks>
/// This is the core FSM for JSON constrained decoding. It processes one character
/// at a time via <see cref="TryAdvance"/> and reports validity/completion.
/// The nesting stack tracks object vs array contexts for proper brace/bracket matching.
/// Fully unmanaged struct — zero heap allocations. Copies by value for cloning.
/// </remarks>
internal struct JsonCharParser
{
    private const int MaxDepth = 64;

    private JsonParserState _state;
    private NestingStack _stack;
    private int _depth;
    private LiteralKind _literalKind;
    private int _literalIndex; // low bits: counter/hex progress. Bit 8: KeyStringFlag

    /// <summary>Creates a new parser in the initial state.</summary>
    public JsonCharParser()
    {
        _state = JsonParserState.Start;
        _stack = default;
        _depth = 0;
        _literalKind = LiteralKind.None;
        _literalIndex = 0;
    }

    /// <summary>
    /// Whether the parser has consumed a complete, valid JSON value
    /// and is at nesting depth 0 (only EOS allowed after this).
    /// </summary>
    public readonly bool IsComplete => _state == JsonParserState.Done;

    /// <summary>
    /// Whether the parser is in a state where the consumed input so far
    /// could be extended to form valid JSON. Used for number termination
    /// checks — numbers are valid at certain intermediate states.
    /// </summary>
    public readonly bool CanTerminateValue => _state is
        JsonParserState.InNumberZero or
        JsonParserState.InNumberIntDigits or
        JsonParserState.InNumberFracDigits or
        JsonParserState.InNumberExpDigits;

    /// <summary>
    /// Returns a hash key representing the effective parser state for mask caching.
    /// Two parser states with the same key will allow the same set of next characters.
    /// Includes literal kind/index and string substate (key vs value, unicode progress)
    /// to avoid cache collisions between states that allow different token sets.
    /// </summary>
    public readonly int GetEffectiveStateKey()
    {
        int stateVal = (int)_state;
        int depthBucket = Math.Min(_depth, 2); // 0, 1, 2+ all behave the same for masking
        int topContext = _depth > 0 ? (int)_stack[_depth - 1] + 1 : 0;
        return HashCode.Combine(stateVal, depthBucket, topContext, (int)_literalKind, _literalIndex);
    }

    /// <summary>
    /// Attempts to advance the parser by one character.
    /// Returns <c>true</c> if the character is valid at the current state;
    /// <c>false</c> if the character would produce invalid JSON.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryAdvance(char c)
    {
        return _state switch
        {
            JsonParserState.Start => TryStart(c),
            JsonParserState.ValueStart => TryValueStart(c),
            JsonParserState.ObjectOpen => TryObjectOpen(c),
            JsonParserState.ObjectColon => TryObjectColon(c),
            JsonParserState.ObjectCommaOrClose => TryObjectCommaOrClose(c),
            JsonParserState.ObjectNextKey => TryObjectNextKey(c),
            JsonParserState.ArrayOpen => TryArrayOpen(c),
            JsonParserState.ArrayCommaOrClose => TryArrayCommaOrClose(c),
            JsonParserState.ArrayNextValue => TryArrayNextValue(c),
            JsonParserState.InString => TryInString(c),
            JsonParserState.InStringEscape => TryInStringEscape(c),
            JsonParserState.InStringUnicode => TryInStringUnicode(c),
            JsonParserState.InNumberSign => TryInNumberSign(c),
            JsonParserState.InNumberZero => TryInNumberZero(c),
            JsonParserState.InNumberIntDigits => TryInNumberIntDigits(c),
            JsonParserState.InNumberDot => TryInNumberDot(c),
            JsonParserState.InNumberFracDigits => TryInNumberFracDigits(c),
            JsonParserState.InNumberExp => TryInNumberExp(c),
            JsonParserState.InNumberExpSign => TryInNumberExpSign(c),
            JsonParserState.InNumberExpDigits => TryInNumberExpDigits(c),
            JsonParserState.InLiteral => TryInLiteral(c),
            JsonParserState.Done => TryDone(c),
            _ => false
        };
    }

    /// <summary>Resets to initial state.</summary>
    public void Reset()
    {
        _state = JsonParserState.Start;
        _depth = 0;
        _literalKind = LiteralKind.None;
        _literalIndex = 0;
    }

    // ── State handlers ──────────────────────────────────────────────

    private bool TryStart(char c)
    {
        // No leading whitespace — go straight to { or [.
        // RFC 8259 allows it, but for generation it wastes tokens
        // and causes models with weak JSON priors to loop on spaces.
        return TryBeginRootValue(c);
    }

    private bool TryBeginRootValue(char c)
    {
        // JSON mode: root must be object or array
        if (c == '{') return PushAndTransition(NestingContext.Object, JsonParserState.ObjectOpen);
        if (c == '[') return PushAndTransition(NestingContext.Array, JsonParserState.ArrayOpen);
        return false;
    }

    private bool TryValueStart(char c)
    {
        if (IsWhitespace(c)) return true;
        return TryBeginValue(c);
    }

    private bool TryBeginValue(char c)
    {
        if (c == '"') { _state = JsonParserState.InString; return true; }
        if (c == '{') return PushAndTransition(NestingContext.Object, JsonParserState.ObjectOpen);
        if (c == '[') return PushAndTransition(NestingContext.Array, JsonParserState.ArrayOpen);
        if (c == '-') { _state = JsonParserState.InNumberSign; return true; }
        if (c == '0') { _state = JsonParserState.InNumberZero; return true; }
        if (c is >= '1' and <= '9') { _state = JsonParserState.InNumberIntDigits; return true; }
        if (c == 't') { _literalKind = LiteralKind.True; _literalIndex = 1; _state = JsonParserState.InLiteral; return true; }
        if (c == 'f') { _literalKind = LiteralKind.False; _literalIndex = 1; _state = JsonParserState.InLiteral; return true; }
        if (c == 'n') { _literalKind = LiteralKind.Null; _literalIndex = 1; _state = JsonParserState.InLiteral; return true; }
        return false;
    }

    // ── Object states ───────────────────────────────────────────────

    private bool TryObjectOpen(char c)
    {
        if (IsWhitespace(c)) return true;
        if (c == '}') { return PopAndTransition(); }
        if (c == '"') { _literalIndex = KeyStringFlag; _state = JsonParserState.InString; return true; }
        return false;
    }

    private bool TryObjectColon(char c)
    {
        if (IsWhitespace(c)) return true;
        if (c == ':') { _state = JsonParserState.ValueStart; return true; }
        return false;
    }

    private bool TryObjectCommaOrClose(char c)
    {
        if (IsWhitespace(c)) return true;
        if (c == ',') { _state = JsonParserState.ObjectNextKey; return true; }
        if (c == '}') { return PopAndTransition(); }
        return false;
    }

    private bool TryObjectNextKey(char c)
    {
        if (IsWhitespace(c)) return true;
        if (c == '"') { _literalIndex = KeyStringFlag; _state = JsonParserState.InString; return true; }
        return false;
    }

    // ── Array states ────────────────────────────────────────────────

    private bool TryArrayOpen(char c)
    {
        if (IsWhitespace(c)) return true;
        if (c == ']') { return PopAndTransition(); }
        return TryBeginValue(c);
    }

    private bool TryArrayCommaOrClose(char c)
    {
        if (IsWhitespace(c)) return true;
        if (c == ',') { _state = JsonParserState.ArrayNextValue; return true; }
        if (c == ']') { return PopAndTransition(); }
        return false;
    }

    private bool TryArrayNextValue(char c)
    {
        if (IsWhitespace(c)) return true;
        return TryBeginValue(c);
    }

    // ── String states ───────────────────────────────────────────────

    private bool TryInString(char c)
    {
        if (c == '\\') { _state = JsonParserState.InStringEscape; return true; }
        if (c == '"')
        {
            TransitionAfterStringClose();
            return true;
        }
        // Any char except control chars (< 0x20) is valid inside a JSON string
        return c >= 0x20;
    }

    private bool TryInStringEscape(char c)
    {
        if (c is '"' or '\\' or '/' or 'b' or 'f' or 'n' or 'r' or 't')
        {
            _state = JsonParserState.InString;
            return true;
        }
        if (c == 'u')
        {
            // Preserve KeyStringFlag, reset hex counter to 0
            _literalIndex = _literalIndex & KeyStringFlag;
            _state = JsonParserState.InStringUnicode;
            return true;
        }
        return false;
    }

    private bool TryInStringUnicode(char c)
    {
        if (IsHexDigit(c))
        {
            _literalIndex++;
            if ((_literalIndex & 0xFF) >= 4)
                _state = JsonParserState.InString;
            return true;
        }
        return false;
    }

    // ── Number states ───────────────────────────────────────────────

    private bool TryInNumberSign(char c)
    {
        if (c == '0') { _state = JsonParserState.InNumberZero; return true; }
        if (c is >= '1' and <= '9') { _state = JsonParserState.InNumberIntDigits; return true; }
        return false;
    }

    private bool TryInNumberZero(char c)
    {
        if (c == '.') { _state = JsonParserState.InNumberDot; return true; }
        if (c is 'e' or 'E') { _state = JsonParserState.InNumberExp; return true; }
        return TryTerminateNumber(c);
    }

    private bool TryInNumberIntDigits(char c)
    {
        if (c is >= '0' and <= '9') return true;
        if (c == '.') { _state = JsonParserState.InNumberDot; return true; }
        if (c is 'e' or 'E') { _state = JsonParserState.InNumberExp; return true; }
        return TryTerminateNumber(c);
    }

    private bool TryInNumberDot(char c)
    {
        if (c is >= '0' and <= '9') { _state = JsonParserState.InNumberFracDigits; return true; }
        return false;
    }

    private bool TryInNumberFracDigits(char c)
    {
        if (c is >= '0' and <= '9') return true;
        if (c is 'e' or 'E') { _state = JsonParserState.InNumberExp; return true; }
        return TryTerminateNumber(c);
    }

    private bool TryInNumberExp(char c)
    {
        if (c is '+' or '-') { _state = JsonParserState.InNumberExpSign; return true; }
        if (c is >= '0' and <= '9') { _state = JsonParserState.InNumberExpDigits; return true; }
        return false;
    }

    private bool TryInNumberExpSign(char c)
    {
        if (c is >= '0' and <= '9') { _state = JsonParserState.InNumberExpDigits; return true; }
        return false;
    }

    private bool TryInNumberExpDigits(char c)
    {
        if (c is >= '0' and <= '9') return true;
        return TryTerminateNumber(c);
    }

    /// <summary>
    /// Handles the implicit termination of a number value when a non-number character appears.
    /// Transitions to the appropriate post-value state, then processes the terminating character.
    /// </summary>
    private bool TryTerminateNumber(char c)
    {
        TransitionAfterValue();
        return TryAdvance(c);
    }

    // ── Literal states ──────────────────────────────────────────────

    private bool TryInLiteral(char c)
    {
        char expected = ExpectedLiteralChar(_literalKind, _literalIndex);
        if (expected != '\0' && c == expected)
        {
            _literalIndex++;
            if (_literalIndex >= LiteralLength(_literalKind))
            {
                _literalKind = LiteralKind.None;
                _literalIndex = 0;
                TransitionAfterValue();
            }
            return true;
        }
        return false;
    }

    // ── Done state ──────────────────────────────────────────────────

    private static bool TryDone(char c)
    {
        // No further characters allowed — constraint forces EOS.
        // RFC 8259 allows trailing ws, but for generation we want to stop immediately.
        return false;
    }

    // ── Helpers ─────────────────────────────────────────────────────

    private void TransitionAfterStringClose()
    {
        if (_depth > 0 && _stack[_depth - 1] == NestingContext.Object)
        {
            // KeyStringFlag is set by ObjectOpen/ObjectNextKey when entering a key string.
            if ((_literalIndex & KeyStringFlag) != 0)
            {
                _literalIndex = 0;
                _state = JsonParserState.ObjectColon;
            }
            else
            {
                _state = JsonParserState.ObjectCommaOrClose;
            }
        }
        else if (_depth > 0 && _stack[_depth - 1] == NestingContext.Array)
        {
            _state = JsonParserState.ArrayCommaOrClose;
        }
        else
        {
            _state = JsonParserState.Done;
        }
    }

    private void TransitionAfterValue()
    {
        if (_depth > 0)
        {
            _state = _stack[_depth - 1] == NestingContext.Object
                ? JsonParserState.ObjectCommaOrClose
                : JsonParserState.ArrayCommaOrClose;
        }
        else
        {
            _state = JsonParserState.Done;
        }
    }

    private bool PopAndTransition()
    {
        if (_depth <= 0) return false;
        _depth--;
        TransitionAfterValue();
        return true;
    }

    private bool PushAndTransition(NestingContext context, JsonParserState nextState)
    {
        if (_depth >= MaxDepth) return false;
        _stack[_depth++] = context;
        _state = nextState;
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsWhitespace(char c) => c is ' ' or '\t' or '\n' or '\r';

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsHexDigit(char c) => c is (>= '0' and <= '9') or (>= 'a' and <= 'f') or (>= 'A' and <= 'F');

    // Flag bit used in _literalIndex to mark that the current string is an object key
    private const int KeyStringFlag = 0x100;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static char ExpectedLiteralChar(LiteralKind kind, int index) => kind switch
    {
        LiteralKind.True => "true"[index],
        LiteralKind.False => "false"[index],
        LiteralKind.Null => "null"[index],
        _ => '\0'
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int LiteralLength(LiteralKind kind) => kind switch
    {
        LiteralKind.True => 4,
        LiteralKind.False => 5,
        LiteralKind.Null => 4,
        _ => 0
    };
}

/// <summary>Parser states for the JSON character-level FSM.</summary>
internal enum JsonParserState : byte
{
    Start,
    ValueStart,
    ObjectOpen,
    ObjectColon,
    ObjectCommaOrClose,
    ObjectNextKey,
    ArrayOpen,
    ArrayCommaOrClose,
    ArrayNextValue,
    InString,
    InStringEscape,
    InStringUnicode,
    InNumberSign,
    InNumberZero,
    InNumberIntDigits,
    InNumberDot,
    InNumberFracDigits,
    InNumberExp,
    InNumberExpSign,
    InNumberExpDigits,
    InLiteral,
    Done
}

/// <summary>Nesting context for JSON objects and arrays.</summary>
internal enum NestingContext : byte
{
    Object,
    Array
}

/// <summary>Which JSON literal keyword is being parsed.</summary>
internal enum LiteralKind : byte
{
    None,
    True,
    False,
    Null
}

/// <summary>
/// Inline fixed-size stack for nesting contexts. Zero heap allocations —
/// copies by value when the containing struct is copied.
/// </summary>
[InlineArray(64)]
internal struct NestingStack
{
    private NestingContext _element;
}
