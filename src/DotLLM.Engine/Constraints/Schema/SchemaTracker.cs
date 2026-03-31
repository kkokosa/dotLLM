using System.Runtime.CompilerServices;

namespace DotLLM.Engine.Constraints.Schema;

/// <summary>
/// Tracks schema position during JSON generation. Advances in lockstep with
/// <see cref="JsonCharParser"/>, observing structural events to enforce schema constraints.
/// </summary>
/// <remarks>
/// Value type — copies by value for zero-alloc cloning. Uses <c>InlineArray</c>
/// for all stacks. Maximum nesting depth: 64 (matches <see cref="JsonCharParser"/>).
/// </remarks>
internal struct SchemaTracker
{
    private const int MaxDepth = 64;
    private const int MaxKeyLength = 128;

    private readonly CompiledSchema _schema;

    // Schema node index stack — parallel to parser's nesting stack.
    // Each entry is the schema node index for the containing object/array.
    private SchemaNodeIdxStack _nodeStack;
    private int _stackDepth;

    // Current schema node index for the value being generated.
    private int _currentNodeIndex;

    // Emitted property bitmask per object nesting level.
    private PropertyBitStack _emittedProps;

    // Key character buffer for matching property names after key string closes.
    private KeyCharBuffer _keyBuffer;
    private int _keyLength;

    // Trie position during key string generation.
    private int _trieNodeIndex;

    // Array item index per array nesting level (for future minItems/maxItems).
    private ArrayIndexStack _arrayIndices;

    // Enum/const trie position during value string generation.
    private int _enumTrieNodeIndex;

    // Track whether we are inside a key string (set on entry, cleared on exit).
    private bool _inKeyString;

    // Track whether we are inside a value string with enum/const constraint.
    private bool _inEnumString;

    /// <summary>
    /// Creates a new schema tracker for the given compiled schema.
    /// </summary>
    /// <param name="schema">The compiled schema (immutable, shared).</param>
    public SchemaTracker(CompiledSchema schema)
    {
        _schema = schema;
        _currentNodeIndex = 0; // root node
        _stackDepth = 0;
        _keyLength = 0;
        _trieNodeIndex = 0;
        _enumTrieNodeIndex = 0;
        _inKeyString = false;
        _inEnumString = false;
    }

    /// <summary>
    /// Whether the schema is fully satisfied.
    /// </summary>
    public readonly bool IsComplete(in JsonCharParser parser) => parser.IsComplete;

    /// <summary>
    /// Checks if a character is allowed by the schema at the current position.
    /// Called BEFORE <see cref="JsonCharParser.TryAdvance"/>.
    /// </summary>
    /// <param name="c">The character to check.</param>
    /// <param name="parser">The current parser state (before advancing).</param>
    /// <returns>True if the schema allows this character.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool IsCharAllowedBySchema(char c, in JsonCharParser parser)
    {
        var state = parser.State;

        return state switch
        {
            JsonParserState.Start => IsValueStartCharAllowed(c, _currentNodeIndex),
            JsonParserState.ValueStart => IsValueStartCharAllowed(c, _currentNodeIndex) || IsWhitespace(c),
            JsonParserState.ObjectOpen => IsObjectOpenCharAllowed(c),
            JsonParserState.ObjectNextKey => IsObjectNextKeyCharAllowed(c),
            JsonParserState.ObjectCommaOrClose => IsObjectCommaOrCloseCharAllowed(c),
            JsonParserState.ObjectColon => true, // parser handles syntax
            JsonParserState.InString => IsInStringCharAllowed(c, parser),
            JsonParserState.InStringEscape => true, // parser handles escape validation
            JsonParserState.InStringUnicode => true, // parser handles hex validation
            JsonParserState.InNumberSign => true,
            JsonParserState.InNumberZero => IsNumberContinuationAllowed(c),
            JsonParserState.InNumberIntDigits => IsNumberContinuationAllowed(c),
            JsonParserState.InNumberDot => true,
            JsonParserState.InNumberFracDigits => IsNumberContinuationAllowed(c),
            JsonParserState.InNumberExp => true,
            JsonParserState.InNumberExpSign => true,
            JsonParserState.InNumberExpDigits => IsNumberContinuationAllowed(c),
            JsonParserState.InLiteral => true, // parser validates literal chars
            JsonParserState.ArrayOpen => IsArrayOpenCharAllowed(c),
            JsonParserState.ArrayCommaOrClose => true, // parser handles syntax
            JsonParserState.ArrayNextValue => IsValueStartCharAllowed(c, GetArrayItemNodeIndex()) || IsWhitespace(c),
            JsonParserState.Done => true, // parser rejects everything at Done
            _ => true,
        };
    }

    /// <summary>
    /// Called AFTER a character has been successfully accepted by the JSON parser.
    /// Detects structural events and updates schema position accordingly.
    /// </summary>
    /// <param name="c">The character that was just accepted.</param>
    /// <param name="parser">The parser state AFTER advancing.</param>
    public void OnCharAdvanced(char c, in JsonCharParser parser)
    {
        var newState = parser.State;

        // Object opened: '{' → parser is now in ObjectOpen
        if (c == '{' && newState == JsonParserState.ObjectOpen)
        {
            PushObject();
            return;
        }

        // Array opened: '[' → parser is now in ArrayOpen
        if (c == '[' && newState == JsonParserState.ArrayOpen)
        {
            PushArray();
            return;
        }

        // Key string started: '"' in ObjectOpen/ObjectNextKey → parser moves to InString
        if (c == '"' && newState == JsonParserState.InString && parser.IsKeyString)
        {
            StartKeyString();
            return;
        }

        // Value string started: '"' when not key → parser moves to InString
        if (c == '"' && newState == JsonParserState.InString && !parser.IsKeyString && !_inKeyString)
        {
            StartValueString();
            return;
        }

        // Character inside key string
        if (_inKeyString && newState == JsonParserState.InString)
        {
            AppendKeyChar(c);
            return;
        }

        // Character inside enum/const value string
        if (_inEnumString && newState == JsonParserState.InString)
        {
            AdvanceEnumTrie(c);
            return;
        }

        // Key string closed: parser transitioned out of InString to ObjectColon
        if (_inKeyString && newState == JsonParserState.ObjectColon)
        {
            FinishKeyString();
            return;
        }

        // Value string closed (non-key)
        if (_inEnumString && newState != JsonParserState.InString &&
            newState != JsonParserState.InStringEscape && newState != JsonParserState.InStringUnicode)
        {
            _inEnumString = false;
            // Value complete at current level — handled by PopIfValueComplete below
        }

        // Object closed: '}' → depth decreased
        if (c == '}' && parser.Depth < _stackDepth)
        {
            PopObject();
            return;
        }

        // Array closed: ']' → depth decreased
        if (c == ']' && parser.Depth < _stackDepth)
        {
            PopArray();
            return;
        }

        // Array comma: ',' in array context → next item
        if (c == ',' && newState == JsonParserState.ArrayNextValue)
        {
            AdvanceArrayItem();
            return;
        }

        // Value complete in container context — restore _currentNodeIndex to parent container.
        // After FinishKeyString sets _currentNodeIndex to a property's value node, we need to
        // restore it to the parent object node when the value finishes (string closes, number
        // terminates + comma, literal completes, etc.). Container close (PopObject/PopArray)
        // and array comma (AdvanceArrayItem) have their own return paths above.
        if (_stackDepth > 0 &&
            newState is JsonParserState.ObjectCommaOrClose
                     or JsonParserState.ObjectNextKey
                     or JsonParserState.ArrayCommaOrClose)
        {
            _currentNodeIndex = _nodeStack[_stackDepth - 1];
        }
    }

    /// <summary>
    /// Returns a composite state key for mask caching incorporating schema position.
    /// Uses a struct key to avoid hash collisions (all fields compared exactly).
    /// </summary>
    public readonly SchemaStateKey GetSchemaStateKey(in JsonCharParser parser)
    {
        int parserKey = parser.GetEffectiveStateKey();
        ulong emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0;
        int triePos = _inKeyString ? _trieNodeIndex : (_inEnumString ? _enumTrieNodeIndex : 0);

        return new SchemaStateKey(parserKey, _currentNodeIndex, emitted, triePos);
    }

    /// <summary>Resets to initial state.</summary>
    public void Reset()
    {
        _currentNodeIndex = 0;
        _stackDepth = 0;
        _keyLength = 0;
        _trieNodeIndex = 0;
        _enumTrieNodeIndex = 0;
        _inKeyString = false;
        _inEnumString = false;
    }

    // ── Value start type restriction ────────────────────────────────

    private readonly bool IsValueStartCharAllowed(char c, int nodeIndex)
    {
        if (IsWhitespace(c))
            return true;

        ref readonly var node = ref GetNode(nodeIndex);
        var types = node.AllowedTypes;

        // If anyOf, merge types from all alternatives.
        // TODO: This is an overapproximation — after the first character disambiguates the
        // branch, nested constraints (required keys, enums, object shapes) from each branch
        // are not enforced. Full branch narrowing requires parallel tracker states.
        if (node.AnyOfNodeIndices != null)
        {
            types = JsonSchemaType.None;
            foreach (int altIdx in node.AnyOfNodeIndices)
                types |= GetNode(altIdx).AllowedTypes;
        }

        return c switch
        {
            '{' => types.HasFlag(JsonSchemaType.Object),
            '[' => types.HasFlag(JsonSchemaType.Array),
            '"' => types.HasFlag(JsonSchemaType.String),
            '-' => types.HasFlag(JsonSchemaType.Number) || types.HasFlag(JsonSchemaType.Integer),
            >= '0' and <= '9' => types.HasFlag(JsonSchemaType.Number) || types.HasFlag(JsonSchemaType.Integer),
            't' or 'f' => types.HasFlag(JsonSchemaType.Boolean),
            'n' => types.HasFlag(JsonSchemaType.Null),
            _ => false,
        };
    }

    // ── Object state restrictions ───────────────────────────────────

    private readonly bool IsObjectOpenCharAllowed(char c)
    {
        if (IsWhitespace(c))
            return true;

        ref readonly var node = ref GetNode(_currentNodeIndex);

        if (c == '}')
        {
            // Can close empty object only if no required properties
            return node.RequiredBitmask == 0;
        }

        if (c == '"')
        {
            // Start key — must have properties defined (or allow additional)
            return node.Properties != null || !node.AdditionalPropertiesForbidden;
        }

        return false;
    }

    private readonly bool IsObjectNextKeyCharAllowed(char c)
    {
        if (IsWhitespace(c))
            return true;

        if (c == '"')
        {
            ref readonly var node = ref GetNode(_currentNodeIndex);
            // Must have remaining properties or allow additional
            if (node.AdditionalPropertiesForbidden && node.PropertyNames != null)
            {
                ulong emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0;
                ulong allProps = node.PropertyNames.Length < 64
                    ? (1UL << node.PropertyNames.Length) - 1
                    : ~0UL;
                // If all properties emitted, no more keys allowed
                return (allProps & ~emitted) != 0;
            }
            return true;
        }

        return false;
    }

    private readonly bool IsObjectCommaOrCloseCharAllowed(char c)
    {
        if (IsWhitespace(c))
            return true;

        if (c == '}')
        {
            // Can close only if all required properties emitted
            ref readonly var node = ref GetNode(_currentNodeIndex);
            ulong emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0;
            return (node.RequiredBitmask & ~emitted) == 0;
        }

        if (c == ',')
        {
            // Can continue if there are more properties possible
            ref readonly var node = ref GetNode(_currentNodeIndex);
            if (node.AdditionalPropertiesForbidden && node.PropertyNames != null)
            {
                ulong emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0;
                ulong allProps = node.PropertyNames.Length < 64
                    ? (1UL << node.PropertyNames.Length) - 1
                    : ~0UL;
                return (allProps & ~emitted) != 0;
            }
            return true;
        }

        return false;
    }

    // ── Array state restrictions ────────────────────────────────────

    private readonly bool IsArrayOpenCharAllowed(char c)
    {
        if (IsWhitespace(c))
            return true;

        if (c == ']')
            return true; // empty array always allowed (no minItems in MVP)

        // First item — check items schema type
        int itemsNode = GetArrayItemNodeIndex();
        if (itemsNode >= 0)
            return IsValueStartCharAllowed(c, itemsNode);

        return true; // unconstrained items
    }

    // ── String content restrictions ─────────────────────────────────

    private readonly bool IsInStringCharAllowed(char c, in JsonCharParser parser)
    {
        // Key string: restrict to trie
        if (_inKeyString)
        {
            if (c == '"')
            {
                // Closing quote — property name must be complete (terminal)
                return _schema.PropertyTries.Length > 0 && IsTrieTerminal();
            }
            if (c == '\\')
                return true; // escape sequences handled by parser
            return IsTrieCharValid(c);
        }

        // Enum/const value string: restrict to enum trie
        if (_inEnumString)
        {
            if (c == '"')
            {
                // Closing quote — value must be complete
                return IsEnumTrieTerminal();
            }
            if (c == '\\')
                return true;
            return IsEnumTrieCharValid(c);
        }

        // Unconstrained string — parser handles
        return true;
    }

    // ── Number restrictions (integer type) ──────────────────────────

    private readonly bool IsNumberContinuationAllowed(char c)
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        var types = node.AllowedTypes;

        // If only Integer (no Number), reject fractional/exponent parts
        if (types.HasFlag(JsonSchemaType.Integer) && !types.HasFlag(JsonSchemaType.Number))
        {
            if (c is '.' or 'e' or 'E')
                return false;
        }

        return true;
    }

    // ── Stack operations ────────────────────────────────────────────

    private void PushObject()
    {
        if (_stackDepth >= MaxDepth) return;
        _nodeStack[_stackDepth] = _currentNodeIndex;
        _emittedProps[_stackDepth] = 0;
        _stackDepth++;
    }

    private void PushArray()
    {
        if (_stackDepth >= MaxDepth) return;
        _nodeStack[_stackDepth] = _currentNodeIndex;
        _arrayIndices[_stackDepth] = 0;
        _stackDepth++;

        // Set current node to items schema for first element
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.ItemsNodeIndex >= 0)
            _currentNodeIndex = node.ItemsNodeIndex;
    }

    private void PopObject()
    {
        if (_stackDepth <= 0) return;
        _stackDepth--;
        // Restore parent's current node (will be set by parent context)
        if (_stackDepth > 0)
        {
            _currentNodeIndex = _nodeStack[_stackDepth - 1];
        }
        else
        {
            _currentNodeIndex = 0; // back to root
        }
    }

    private void PopArray()
    {
        if (_stackDepth <= 0) return;
        _stackDepth--;
        if (_stackDepth > 0)
        {
            _currentNodeIndex = _nodeStack[_stackDepth - 1];
        }
        else
        {
            _currentNodeIndex = 0;
        }
    }

    private void AdvanceArrayItem()
    {
        if (_stackDepth > 0)
        {
            _arrayIndices[_stackDepth - 1]++;
            // Reset current node to items schema for next element
            int parentNode = _nodeStack[_stackDepth - 1];
            ref readonly var node = ref GetNode(parentNode);
            if (node.ItemsNodeIndex >= 0)
                _currentNodeIndex = node.ItemsNodeIndex;
        }
    }

    // ── Key string tracking ─────────────────────────────────────────

    private void StartKeyString()
    {
        _inKeyString = true;
        _keyLength = 0;
        _trieNodeIndex = 0; // root of property name trie

        // Resolve the property trie for the current object
        // _currentNodeIndex should be the object node (top of stack)
    }

    private void AppendKeyChar(char c)
    {
        if (_keyLength < MaxKeyLength)
            _keyBuffer[_keyLength++] = c;

        // Advance trie
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.PropertyTrieIndex >= 0)
        {
            var trie = _schema.PropertyTries[node.PropertyTrieIndex];
            if (trie.TryGetChild(_trieNodeIndex, c, out int child))
                _trieNodeIndex = child;
        }
    }

    private void FinishKeyString()
    {
        _inKeyString = false;

        // Build key name from buffer (inline to avoid ref-escape issues with InlineArray)
        string keyName = new(((ReadOnlySpan<char>)_keyBuffer)[.._keyLength]);

        // Look up property in schema and set current node to the property's value schema
        ref readonly var objectNode = ref GetNode(_currentNodeIndex);
        if (objectNode.Properties != null && objectNode.Properties.TryGetValue(keyName, out int valueNodeIndex))
        {
            // Mark property as emitted
            if (objectNode.PropertyNames != null)
            {
                int bitPos = Array.IndexOf(objectNode.PropertyNames, keyName);
                if (bitPos >= 0 && bitPos < 64 && _stackDepth > 0)
                    _emittedProps[_stackDepth - 1] |= 1UL << bitPos;
            }

            _currentNodeIndex = valueNodeIndex;
        }
        // If property not in schema and additionalProperties allowed, keep unconstrained
    }

    // ── Value string (enum/const) tracking ──────────────────────────

    // TODO: Non-string enum/const values (e.g. {"const":1}, {"enum":[true,false]}) are not
    // character-level constrained — only type restriction applies. Full enforcement requires
    // character-sequence matching for literals/numbers, a fundamentally different mechanism.
    private void StartValueString()
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.EnumTrieIndex >= 0)
        {
            _inEnumString = true;
            _enumTrieNodeIndex = 0; // root of enum trie
        }
    }

    private void AdvanceEnumTrie(char c)
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.EnumTrieIndex >= 0)
        {
            var trie = _schema.PropertyTries[node.EnumTrieIndex];
            if (trie.TryGetChild(_enumTrieNodeIndex, c, out int child))
                _enumTrieNodeIndex = child;
        }
    }

    // ── Trie helpers ────────────────────────────────────────────────

    private readonly bool IsTrieCharValid(char c)
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.PropertyTrieIndex < 0) return true; // no trie = unconstrained

        var trie = _schema.PropertyTries[node.PropertyTrieIndex];

        // Also filter to only non-emitted properties
        if (node.AdditionalPropertiesForbidden)
            return trie.TryGetChild(_trieNodeIndex, c, out _);

        // If additional properties allowed, any char is valid even if not in trie
        return trie.TryGetChild(_trieNodeIndex, c, out _) || !node.AdditionalPropertiesForbidden;
    }

    private readonly bool IsTrieTerminal()
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.PropertyTrieIndex < 0) return true;

        var trie = _schema.PropertyTries[node.PropertyTrieIndex];

        // Terminal in trie = complete property name
        if (!trie.IsTerminal(_trieNodeIndex))
        {
            // Not a complete property name in the trie
            // If additional properties are allowed, it could still be valid
            return !node.AdditionalPropertiesForbidden;
        }

        // Check it's not already emitted (only if additionalProperties is forbidden,
        // otherwise duplicates are technically valid JSON though unusual)
        if (node.AdditionalPropertiesForbidden)
        {
            string? name = trie.GetCompleteName(_trieNodeIndex);
            if (name != null && node.PropertyNames != null)
            {
                int bitPos = Array.IndexOf(node.PropertyNames, name);
                if (bitPos >= 0 && bitPos < 64 && _stackDepth > 0)
                {
                    ulong emitted = _emittedProps[_stackDepth - 1];
                    if ((emitted & (1UL << bitPos)) != 0)
                        return false; // already emitted
                }
            }
        }

        return true;
    }

    private readonly bool IsEnumTrieCharValid(char c)
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.EnumTrieIndex < 0) return true;

        var trie = _schema.PropertyTries[node.EnumTrieIndex];
        return trie.TryGetChild(_enumTrieNodeIndex, c, out _);
    }

    private readonly bool IsEnumTrieTerminal()
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.EnumTrieIndex < 0) return true;

        var trie = _schema.PropertyTries[node.EnumTrieIndex];
        return trie.IsTerminal(_enumTrieNodeIndex);
    }

    private readonly int GetArrayItemNodeIndex()
    {
        if (_stackDepth <= 0) return -1;
        int parentNode = _nodeStack[_stackDepth - 1];
        ref readonly var node = ref GetNode(parentNode);
        return node.ItemsNodeIndex;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private readonly ref readonly SchemaNode GetNode(int index) => ref _schema.Nodes[index];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsWhitespace(char c) => c is ' ' or '\t' or '\n' or '\r';
}

/// <summary>InlineArray for schema node index stack.</summary>
[InlineArray(64)]
internal struct SchemaNodeIdxStack
{
    private int _element;
}

/// <summary>InlineArray for emitted property bitmask stack.</summary>
[InlineArray(64)]
internal struct PropertyBitStack
{
    private ulong _element;
}

/// <summary>InlineArray for key character buffer.</summary>
[InlineArray(128)]
internal struct KeyCharBuffer
{
    private char _element;
}

/// <summary>InlineArray for array item index stack.</summary>
[InlineArray(64)]
internal struct ArrayIndexStack
{
    private int _element;
}

/// <summary>
/// Collision-free cache key for schema constraint mask lookup.
/// All fields are compared exactly — no hash compression.
/// </summary>
internal readonly record struct SchemaStateKey(
    int ParserKey,
    int NodeIdx,
    ulong EmittedProps,
    int TriePos);
