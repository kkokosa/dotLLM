using DotLLM.Core.Tensors;
using DotLLM.Ascend.Interop;

namespace DotLLM.Ascend;

public sealed partial class AscendSession
{
    /// <summary>Converts FP16 and FP32 storage on the NPU without host round trips.</summary>
    public void Cast(AscendTensor input, AscendTensor output)
    {
        ArgumentNullException.ThrowIfNull(input); ArgumentNullException.ThrowIfNull(output);
        lock (_gate)
        {
            Validate(input); Validate(output);
            if (input.StorageShape != output.StorageShape || input.DType == output.DType ||
                ReferenceEquals(input, output)) throw new ArgumentException("Cast requires matching shapes and different floating dtypes.");
            MakeCurrent();
            CheckKernel(AscendNative.Cast(input.DataPointer, output.DataPointer, input.ElementCount,
                output.DType == DType.Float32 ? 1 : 0, _stream), "Cast");
        }
    }

    /// <summary>FP32 Linear with resident FP32 weights and KEEP_DTYPE FP32 computation.</summary>
    public void LinearFloat32(AscendTensor input, AscendTensor weight, AscendTensor output)
    {
        ArgumentNullException.ThrowIfNull(weight);
        lock (_gate)
        {
            ValidateMatrix(input, DType.Float32); ValidateMatrix(weight, DType.Float32); ValidateMatrix(output, DType.Float32);
            var x = input.StorageShape; var w = weight.StorageShape; var y = output.StorageShape;
            if (x[1] != w[1] || y[0] != x[0] || y[1] != w[0] || ReferenceEquals(output, input) || ReferenceEquals(output, weight))
                throw new ArgumentException("Linear requires X[M,K], W[N,K], Y[M,N] and separate output.");
            MakeCurrent();
            CheckKernel(AscendNative.LinearF32(input.DataPointer, weight.DataPointer, output.DataPointer,
                x[0], w[0], x[1], _stream), "LinearF32");
        }
    }

    /// <summary>Downloads an FP32 tensor synchronously.</summary>
    public unsafe void Download(AscendTensor tensor, Span<float> values)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        lock (_gate)
        {
            Validate(tensor); RequireType(tensor, DType.Float32);
            if (values.Length != tensor.ElementCount) throw new ArgumentException("Download size differs.");
            MakeCurrent();
            fixed (float* ptr = values)
                AscendNative.Check(AscendNative.aclrtMemcpy((nint)ptr, (nuint)tensor.ByteCount,
                    tensor.DataPointer, (nuint)tensor.ByteCount, 2), "aclrtMemcpy(D2H FP32)");
        }
    }

    /// <summary>Uploads an FP32 tensor synchronously.</summary>
    public unsafe void Upload(AscendTensor tensor, ReadOnlySpan<float> values)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        lock (_gate)
        {
            Validate(tensor); RequireType(tensor, DType.Float32);
            if (values.Length != tensor.ElementCount) throw new ArgumentException("Upload size differs.");
            MakeCurrent();
            fixed (float* ptr = values)
                AscendNative.Check(AscendNative.aclrtMemcpy(tensor.DataPointer, (nuint)tensor.ByteCount,
                    (nint)ptr, (nuint)tensor.ByteCount, 1), "aclrtMemcpy(H2D FP32)");
        }
    }

    /// <summary>Uploads a bounded contiguous FP32 slice.</summary>
    public unsafe void UploadSlice(AscendTensor tensor, long offset, ReadOnlySpan<float> values)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        lock (_gate)
        {
            Validate(tensor); RequireType(tensor, DType.Float32); ValidateRange(tensor, offset, values.Length);
            MakeCurrent(); if (values.IsEmpty) return;
            fixed (float* ptr = values)
                AscendNative.Check(AscendNative.aclrtMemcpy(tensor.DataPointer + checked((nint)(offset * 4)), checked((nuint)(values.Length * 4L)),
                    (nint)ptr, checked((nuint)(values.Length * 4L)), 1), "aclrtMemcpy(H2D FP32 slice)");
        }
    }

    /// <summary>Copies a nonoverlapping FP32 slice within this session.</summary>
    public void CopyFloat32(AscendTensor source, long sourceOffset, AscendTensor destination, long destinationOffset, long count)
    {
        ArgumentNullException.ThrowIfNull(source); ArgumentNullException.ThrowIfNull(destination);
        lock (_gate)
        {
            Validate(source); Validate(destination); RequireType(source, DType.Float32); RequireType(destination, DType.Float32);
            ValidateRange(source, sourceOffset, count); ValidateRange(destination, destinationOffset, count);
            if (ReferenceEquals(source, destination) && count != 0 && sourceOffset < destinationOffset + count && destinationOffset < sourceOffset + count)
                throw new ArgumentException("Overlapping device copies are unsupported.");
            MakeCurrent(); if (count == 0) return;
            AscendNative.Check(AscendNative.aclrtMemcpy(destination.DataPointer + checked((nint)(destinationOffset * 4)), checked((nuint)(count * 4)),
                source.DataPointer + checked((nint)(sourceOffset * 4)), checked((nuint)(count * 4)), 3), "aclrtMemcpy(D2D FP32)");
        }
    }

    /// <summary>Executes RmsNorm using FP32 activations and explicit FP16 weight/table/cache conversion.</summary>
    public void RmsNormFloat32(AscendTensor input, AscendTensor weight, AscendTensor output, double epsilon)
    {
        ArgumentNullException.ThrowIfNull(weight);
        if (!double.IsFinite(epsilon) || epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon));
        lock (_gate)
        {
            ValidateMatrix(input, DType.Float32);
            Validate(weight);
            RequireType(weight, DType.Float32);
            ValidateOutput(input, output);
            if (weight.StorageShape.Rank != 1 || weight.ElementCount != input.StorageShape[1] ||
                ReferenceEquals(weight, output))
                throw new ArgumentException("RMSNorm weight must match the row width.", nameof(weight));
            MakeCurrent();
            CheckKernel(AscendNative.RmsNormF32(input.DataPointer, weight.DataPointer, output.DataPointer,
                input.StorageShape[0], input.StorageShape[1], epsilon, _stream), "RmsNormF32");
        }
    }

    /// <summary>Executes Add using FP32 activations and explicit FP16 weight/table/cache conversion.</summary>
    public void AddFloat32(AscendTensor left, AscendTensor right, AscendTensor output)
    {
        ArgumentNullException.ThrowIfNull(right);
        lock (_gate)
        {
            ValidateMatrix(left, DType.Float32);
            Validate(right);
            ValidateOutput(left, output);
            var shape = right.StorageShape;
            int rightRows;
            if (shape.Rank == 1 && shape[0] == left.StorageShape[1]) rightRows = 1;
            else if (shape.Rank == 2 && shape[1] == left.StorageShape[1] &&
                (shape[0] == 1 || shape[0] == left.StorageShape[0])) rightRows = shape[0];
            else throw new ArgumentException("Right operand must have the same shape or be a row bias.", nameof(right));
            if (ReferenceEquals(right, output)) throw new ArgumentException("Output cannot alias the bias.", nameof(output));
            MakeCurrent();
            CheckKernel(AscendNative.AddF32(left.DataPointer, right.DataPointer, output.DataPointer,
                left.StorageShape[0], left.StorageShape[1], rightRows, right.DType == DType.Float16 ? 1 : 0, _stream), "AddF32");
        }
    }

    /// <summary>Executes SwiGlu using FP32 activations and explicit FP16 weight/table/cache conversion.</summary>
    public void SwiGluFloat32(AscendTensor gate, AscendTensor up, AscendTensor output)
    {
        ArgumentNullException.ThrowIfNull(up);
        lock (_gate)
        {
            ValidateMatrix(gate, DType.Float32);
            Validate(up);
            RequireType(up, DType.Float32);
            ValidateOutput(gate, output);
            if (gate.StorageShape != up.StorageShape || ReferenceEquals(up, output))
                throw new ArgumentException("SwiGLU inputs must share a shape and not alias output.");
            MakeCurrent();
            CheckKernel(AscendNative.SwiGluF32(gate.DataPointer, up.DataPointer, output.DataPointer,
                gate.ElementCount, _stream), "SwiGluF32");
        }
    }

    /// <summary>Executes Rope using FP32 activations and explicit FP16 weight/table/cache conversion.</summary>
    public void RopeFloat32(AscendTensor input, AscendTensor cos, AscendTensor sin, AscendTensor output,
        int heads, int headDim, int rotaryDim, bool neox)
    {
        if (heads <= 0 || headDim <= 0 || rotaryDim <= 0 || rotaryDim > headDim || rotaryDim % 2 != 0)
            throw new ArgumentOutOfRangeException(nameof(rotaryDim));
        lock (_gate)
        {
            ValidateMatrix(input, DType.Float32);
            ValidateMatrix(cos);
            ValidateMatrix(sin);
            ValidateOutput(input, output);
            if (input.StorageShape[1] != checked(heads * headDim) ||
                cos.StorageShape[0] != input.StorageShape[0] || cos.StorageShape[1] != rotaryDim ||
                sin.StorageShape != cos.StorageShape || ReferenceEquals(cos, output) || ReferenceEquals(sin, output))
                throw new ArgumentException("Invalid RoPE input/table shape or alias.");
            MakeCurrent();
            CheckKernel(AscendNative.RopeF32(input.DataPointer, cos.DataPointer, sin.DataPointer, output.DataPointer,
                input.StorageShape[0], heads, headDim, rotaryDim, neox ? 1 : 0, _stream), "RopeF32");
        }
    }

    /// <summary>Executes Attention using FP32 activations and explicit FP16 weight/table/cache conversion.</summary>
    public void AttentionFloat32(AscendTensor query, AscendTensor keys, AscendTensor values,
        AscendTensor output, int heads, int kvHeads, int headDim, int validKvRows, int queryOffset)
    {
        if (heads <= 0 || kvHeads <= 0 || heads % kvHeads != 0 || headDim <= 0 ||
            validKvRows <= 0 || queryOffset < 0)
            throw new ArgumentOutOfRangeException(nameof(heads));
        lock (_gate)
        {
            ValidateMatrix(query, DType.Float32);
            ValidateMatrix(keys, keys.DType);
            ValidateMatrix(values, keys.DType);
            ValidateOutput(query, output);
            if (query.StorageShape[1] != checked(heads * headDim) ||
                keys.StorageShape[1] != checked(kvHeads * headDim) ||
                keys.StorageShape != values.StorageShape || validKvRows > keys.StorageShape[0] ||
                query.StorageShape[0] > validKvRows || queryOffset > validKvRows - query.StorageShape[0] ||
                ReferenceEquals(keys, output) || ReferenceEquals(values, output))
                throw new ArgumentException("Invalid attention shape, visible range, or output alias.");
            MakeCurrent();
            CheckKernel(AscendNative.AttentionF32(query.DataPointer, keys.DataPointer, values.DataPointer,
                output.DataPointer, query.StorageShape[0], validKvRows, heads, kvHeads, headDim, queryOffset, keys.DType == DType.Float16 ? 1 : 0, _stream),
                "AttentionF32");
        }
    }
}
