using System.Runtime.InteropServices;
using DotLLM.Ascend.Interop;

namespace DotLLM.Ascend;

public sealed partial class AscendSession
{
    private void CheckKernel(int status, string operation)
    {
        if (status == 0) return;
        _faulted = true;
        throw new AscendException($"{operation}(device={DeviceId})",
            status, Marshal.PtrToStringUTF8(AscendNative.LastError()));
    }

    private static void RequireType(AscendTensor tensor, DotLLM.Core.Tensors.DType dtype)
    {
        if (tensor.DType != dtype) throw new ArgumentException($"Expected {dtype} tensor.");
    }

    private void ValidateMatrix(AscendTensor tensor) => ValidateMatrix(tensor, DotLLM.Core.Tensors.DType.Float16);

    private void ValidateMatrix(AscendTensor tensor, DotLLM.Core.Tensors.DType dtype)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        Validate(tensor);
        RequireType(tensor, dtype);
        if (tensor.StorageShape.Rank != 2)
            throw new ArgumentException("Expected a rank-two FP16 matrix.", nameof(tensor));
    }

    private void ValidateOutput(AscendTensor input, AscendTensor output)
    {
        ArgumentNullException.ThrowIfNull(output);
        Validate(output);
        RequireType(output, input.DType);
        if (input.StorageShape != output.StorageShape || ReferenceEquals(input, output))
            throw new ArgumentException("Output must have the input shape and separate storage.", nameof(output));
    }

    /// <summary>RMS normalization of each matrix row with a shared FP16 weight vector.</summary>
    public void RmsNormFloat16(AscendTensor input, AscendTensor weight, AscendTensor output, double epsilon)
    {
        ArgumentNullException.ThrowIfNull(weight);
        if (!double.IsFinite(epsilon) || epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon));
        lock (_gate)
        {
            ValidateMatrix(input);
            Validate(weight);
            RequireType(weight, DotLLM.Core.Tensors.DType.Float16);
            ValidateOutput(input, output);
            if (weight.StorageShape.Rank != 1 || weight.ElementCount != input.StorageShape[1] ||
                ReferenceEquals(weight, output))
                throw new ArgumentException("RMSNorm weight must match the row width.", nameof(weight));
            MakeCurrent();
            CheckKernel(AscendNative.RmsNormF16(input.DataPointer, weight.DataPointer, output.DataPointer,
                input.StorageShape[0], input.StorageShape[1], epsilon, _stream), "RmsNormF16");
        }
    }

    /// <summary>Adds matrices, or broadcasts a vector/one-row bias. Output must be separate.</summary>
    public void AddFloat16(AscendTensor left, AscendTensor right, AscendTensor output)
    {
        ArgumentNullException.ThrowIfNull(right);
        lock (_gate)
        {
            ValidateMatrix(left);
            Validate(right);
            RequireType(right, DotLLM.Core.Tensors.DType.Float16);
            ValidateOutput(left, output);
            var shape = right.StorageShape;
            int rightRows;
            if (shape.Rank == 1 && shape[0] == left.StorageShape[1]) rightRows = 1;
            else if (shape.Rank == 2 && shape[1] == left.StorageShape[1] &&
                (shape[0] == 1 || shape[0] == left.StorageShape[0])) rightRows = shape[0];
            else throw new ArgumentException("Right operand must have the same shape or be a row bias.", nameof(right));
            if (ReferenceEquals(right, output)) throw new ArgumentException("Output cannot alias the bias.", nameof(output));
            MakeCurrent();
            CheckKernel(AscendNative.AddF16(left.DataPointer, right.DataPointer, output.DataPointer,
                left.StorageShape[0], left.StorageShape[1], rightRows, _stream), "AddF16");
        }
    }

    /// <summary>Computes SiLU(gate) * up with FP32 intermediates and FP16 output.</summary>
    public void SwiGluFloat16(AscendTensor gate, AscendTensor up, AscendTensor output)
    {
        ArgumentNullException.ThrowIfNull(up);
        lock (_gate)
        {
            ValidateMatrix(gate);
            Validate(up);
            RequireType(up, DotLLM.Core.Tensors.DType.Float16);
            ValidateOutput(gate, output);
            if (gate.StorageShape != up.StorageShape || ReferenceEquals(up, output))
                throw new ArgumentException("SwiGLU inputs must share a shape and not alias output.");
            MakeCurrent();
            CheckKernel(AscendNative.SwiGluF16(gate.DataPointer, up.DataPointer, output.DataPointer,
                gate.ElementCount, _stream), "SwiGluF16");
        }
    }

    /// <summary>
    /// Rotates a [tokens, heads*headDim] matrix using [tokens, rotaryDim] cos/sin tables.
    /// Tables repeat frequencies per pair for interleaved mode, or across halves for NeoX.
    /// Dimensions after rotaryDim are copied unchanged.
    /// </summary>
    public void RopeFloat16(AscendTensor input, AscendTensor cos, AscendTensor sin, AscendTensor output,
        int heads, int headDim, int rotaryDim, bool neox)
    {
        if (heads <= 0 || headDim <= 0 || rotaryDim <= 0 || rotaryDim > headDim || rotaryDim % 2 != 0)
            throw new ArgumentOutOfRangeException(nameof(rotaryDim));
        lock (_gate)
        {
            ValidateMatrix(input);
            ValidateMatrix(cos);
            ValidateMatrix(sin);
            ValidateOutput(input, output);
            if (input.StorageShape[1] != checked(heads * headDim) ||
                cos.StorageShape[0] != input.StorageShape[0] || cos.StorageShape[1] != rotaryDim ||
                sin.StorageShape != cos.StorageShape || ReferenceEquals(cos, output) || ReferenceEquals(sin, output))
                throw new ArgumentException("Invalid RoPE input/table shape or alias.");
            MakeCurrent();
            CheckKernel(AscendNative.RopeF16(input.DataPointer, cos.DataPointer, sin.DataPointer, output.DataPointer,
                input.StorageShape[0], heads, headDim, rotaryDim, neox ? 1 : 0, _stream), "RopeF16");
        }
    }

    /// <summary>
    /// Causal GQA over FP16 Q/K/V, using FP32 scores and accumulation.
    /// Only the first validKvRows of K/V are visible; query row i is at queryOffset+i.
    /// </summary>
    public void AttentionFloat16(AscendTensor query, AscendTensor keys, AscendTensor values,
        AscendTensor output, int heads, int kvHeads, int headDim, int validKvRows, int queryOffset)
    {
        if (heads <= 0 || kvHeads <= 0 || heads % kvHeads != 0 || headDim <= 0 ||
            validKvRows <= 0 || queryOffset < 0)
            throw new ArgumentOutOfRangeException(nameof(heads));
        lock (_gate)
        {
            ValidateMatrix(query);
            ValidateMatrix(keys);
            ValidateMatrix(values);
            ValidateOutput(query, output);
            if (query.StorageShape[1] != checked(heads * headDim) ||
                keys.StorageShape[1] != checked(kvHeads * headDim) ||
                keys.StorageShape != values.StorageShape || validKvRows > keys.StorageShape[0] ||
                query.StorageShape[0] > validKvRows || queryOffset > validKvRows - query.StorageShape[0] ||
                ReferenceEquals(keys, output) || ReferenceEquals(values, output))
                throw new ArgumentException("Invalid attention shape, visible range, or output alias.");
            MakeCurrent();
            CheckKernel(AscendNative.AttentionF16(query.DataPointer, keys.DataPointer, values.DataPointer,
                output.DataPointer, query.StorageShape[0], validKvRows, heads, kvHeads, headDim, queryOffset, _stream),
                "AttentionF16");
        }
    }

    /// <summary>Uploads a contiguous slice. The source remains pinned until the copy completes.</summary>
    public unsafe void UploadSlice(AscendTensor destination, long destinationOffset, ReadOnlySpan<Half> values)
    {
        ArgumentNullException.ThrowIfNull(destination);
        lock (_gate)
        {
            Validate(destination);
            RequireType(destination, DotLLM.Core.Tensors.DType.Float16);
            ValidateRange(destination, destinationOffset, values.Length);
            MakeCurrent();
            if (values.IsEmpty) return;
            fixed (Half* source = values)
                AscendNative.Check(AscendNative.aclrtMemcpy(
                    destination.DataPointer + checked((nint)(destinationOffset * 2)),
                    checked((nuint)(values.Length * 2L)), (nint)source,
                    checked((nuint)(values.Length * 2L)), 1), "aclrtMemcpy(H2D slice)");
        }
    }

    /// <summary>Copies a slice between tensors belonging to this session. Overlap is rejected.</summary>
    public void CopyFloat16(AscendTensor source, long sourceOffset,
        AscendTensor destination, long destinationOffset, long elementCount)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        lock (_gate)
        {
            Validate(source);
            Validate(destination);
            RequireType(source, DotLLM.Core.Tensors.DType.Float16);
            RequireType(destination, DotLLM.Core.Tensors.DType.Float16);
            ValidateRange(source, sourceOffset, elementCount);
            ValidateRange(destination, destinationOffset, elementCount);
            if (ReferenceEquals(source, destination) && elementCount != 0 &&
                sourceOffset < destinationOffset + elementCount && destinationOffset < sourceOffset + elementCount)
                throw new ArgumentException("Overlapping device copies are unsupported.");
            MakeCurrent();
            if (elementCount == 0) return;
            AscendNative.Check(AscendNative.aclrtMemcpy(
                destination.DataPointer + checked((nint)(destinationOffset * 2)), checked((nuint)(elementCount * 2)),
                source.DataPointer + checked((nint)(sourceOffset * 2)), checked((nuint)(elementCount * 2)), 3),
                "aclrtMemcpy(D2D slice)");
        }
    }

    private static void ValidateRange(AscendTensor tensor, long offset, long count)
    {
        if (offset < 0 || count < 0 || count > tensor.ElementCount || offset > tensor.ElementCount - count)
            throw new ArgumentOutOfRangeException(nameof(offset), "Slice exceeds tensor storage.");
    }
}
