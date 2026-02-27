using DotLLM.Core.Tensors;

namespace DotLLM.Core.Backends;

/// <summary>
/// Executes compute kernels on a backend. Implementations provide CPU (SIMD) or GPU (CUDA) kernels.
/// </summary>
public interface IKernelRunner
{
    /// <summary>Matrix multiplication: C = A @ B.</summary>
    /// <param name="a">Left operand.</param>
    /// <param name="b">Right operand.</param>
    /// <param name="result">Output tensor (pre-allocated).</param>
    void MatMul(ITensor a, ITensor b, ITensor result);

    /// <summary>RMS normalization in-place.</summary>
    /// <param name="input">Input tensor to normalize.</param>
    /// <param name="weight">Scale weights.</param>
    /// <param name="epsilon">Normalization epsilon.</param>
    /// <param name="result">Output tensor (pre-allocated).</param>
    void RmsNorm(ITensor input, ITensor weight, float epsilon, ITensor result);

    /// <summary>SiLU activation: x * sigmoid(x).</summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="result">Output tensor (pre-allocated).</param>
    void SiLU(ITensor input, ITensor result);

    /// <summary>Softmax over the last dimension.</summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="result">Output tensor (pre-allocated).</param>
    void Softmax(ITensor input, ITensor result);

    /// <summary>Element-wise addition: result = a + b.</summary>
    /// <param name="a">First operand.</param>
    /// <param name="b">Second operand.</param>
    /// <param name="result">Output tensor (pre-allocated).</param>
    void Add(ITensor a, ITensor b, ITensor result);

    /// <summary>Element-wise multiplication: result = a * b.</summary>
    /// <param name="a">First operand.</param>
    /// <param name="b">Second operand.</param>
    /// <param name="result">Output tensor (pre-allocated).</param>
    void Multiply(ITensor a, ITensor b, ITensor result);
}
