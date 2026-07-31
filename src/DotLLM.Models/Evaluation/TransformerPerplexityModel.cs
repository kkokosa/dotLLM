using DotLLM.Core.Evaluation;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;

namespace DotLLM.Models.Evaluation;

/// <summary>Adapts <see cref="TransformerModel"/> to <see cref="IPerplexityModel"/>.</summary>
/// <remarks>
/// Holds a borrowed reference: the adapter does not own the model and does not dispose it, so the
/// caller keeps a single resident copy of the weights. This is the whole point of the evaluator
/// taking a constructed model rather than a path — on a unified-memory part with a large VRAM
/// carve-out, a second host-side copy is the difference between running and not.
/// </remarks>
public sealed class TransformerPerplexityModel : IPerplexityModel
{
    private readonly TransformerModel _model;
    private readonly int _deviceId;

    /// <param name="model">An already-loaded model. Not owned; not disposed by this adapter.</param>
    /// <param name="deviceId">Device for the forward pass; <c>-1</c> is CPU.</param>
    public TransformerPerplexityModel(TransformerModel model, int deviceId = -1)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _deviceId = deviceId;
    }

    /// <inheritdoc/>
    public int VocabSize => _model.Config.VocabSize;

    /// <inheritdoc/>
    public int MaxContextLength => _model.Config.MaxSequenceLength;

    /// <inheritdoc/>
    /// <remarks>
    /// <see cref="TransformerModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int)"/> is
    /// documented as returning logits of shape <c>[seqLen, vocab_size]</c> for all input positions.
    /// </remarks>
    public bool ReturnsAllRows => true;

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions)
        => _model.Forward(tokens, positions, _deviceId);
}
