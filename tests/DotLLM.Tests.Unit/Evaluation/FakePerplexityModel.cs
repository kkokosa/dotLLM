using DotLLM.Core.Evaluation;
using DotLLM.Core.Tensors;

namespace DotLLM.Tests.Unit.Evaluation;

/// <summary>
/// Deterministic <see cref="IPerplexityModel"/> for evaluator tests. Records every window it is
/// asked to score so tests can assert on window tiling, not just on the resulting number.
/// </summary>
internal sealed class FakePerplexityModel : IPerplexityModel, IDisposable
{
    private readonly Func<int, int, float[]> _rowFactory;   // (absolutePosition, vocabSize) => logits
    private readonly List<int[]> _forwardCalls = [];
    private readonly List<UnmanagedTensor> _issued = [];

    public FakePerplexityModel(
        int vocabSize, int maxContextLength, bool returnsAllRows,
        Func<int, int, float[]> rowFactory)
    {
        VocabSize = vocabSize;
        MaxContextLength = maxContextLength;
        ReturnsAllRows = returnsAllRows;
        _rowFactory = rowFactory;
    }

    public int VocabSize { get; }
    public int MaxContextLength { get; }
    public bool ReturnsAllRows { get; }

    /// <summary>Token windows passed to <see cref="Forward"/>, in call order.</summary>
    public IReadOnlyList<int[]> ForwardCalls => _forwardCalls;

    public unsafe ITensor Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions)
    {
        _forwardCalls.Add(tokens.ToArray());

        int rows = ReturnsAllRows ? tokens.Length : 1;
        int firstRow = ReturnsAllRows ? 0 : tokens.Length - 1;
        var tensor = UnmanagedTensor.Allocate(new TensorShape(rows, VocabSize), DType.Float32);

        var dest = new Span<float>((void*)tensor.DataPointer, rows * VocabSize);
        for (int r = 0; r < rows; r++)
            _rowFactory(positions[firstRow + r], VocabSize).CopyTo(dest[(r * VocabSize)..]);

        _issued.Add(tensor);
        return tensor;
    }

    /// <summary>Uniform logits: every target scores exactly -log(vocabSize).</summary>
    public static Func<int, int, float[]> Uniform => (_, vocab) => new float[vocab];

    public void Dispose()
    {
        foreach (var t in _issued) t.Dispose();
        _issued.Clear();
    }
}
