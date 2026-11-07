using System.Runtime.InteropServices;
using DotLLM.Ascend.Interop;

namespace DotLLM.Ascend;

public sealed partial class AscendSession
{
    /// <summary>Uploads an entire FP16 tensor; the host span can be released on return.</summary>
    public unsafe void Upload(AscendTensor tensor, ReadOnlySpan<Half> values)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        lock (_gate)
        {
            Validate(tensor);
            RequireType(tensor, DotLLM.Core.Tensors.DType.Float16);
            if (values.Length != tensor.ElementCount)
                throw new ArgumentException("Upload size must equal the tensor size.", nameof(values));
            MakeCurrent();
            fixed (Half* hostPtr = values)
                AscendNative.Check(AscendNative.aclrtMemcpy(tensor.DataPointer, (nuint)tensor.ByteCount,
                    (nint)hostPtr, (nuint)tensor.ByteCount, 1), "aclrtMemcpy(H2D)");
        }
    }

    /// <summary>Downloads an entire FP16 tensor into the supplied host span.</summary>
    public unsafe void Download(AscendTensor tensor, Span<Half> values)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        lock (_gate)
        {
            Validate(tensor);
            RequireType(tensor, DotLLM.Core.Tensors.DType.Float16);
            if (values.Length != tensor.ElementCount)
                throw new ArgumentException("Download size must equal the tensor size.", nameof(values));
            MakeCurrent();
            fixed (Half* hostPtr = values)
                AscendNative.Check(AscendNative.aclrtMemcpy((nint)hostPtr, (nuint)tensor.ByteCount,
                    tensor.DataPointer, (nuint)tensor.ByteCount, 2), "aclrtMemcpy(D2H)");
        }
    }

    /// <summary>
    /// Computes Y[M,N] = X[M,K] * W[N,K]^T with FP16 storage and KEEP_DTYPE math.
    /// Tensors must be contiguous and belong to this session. Output cannot alias
    /// either input. Returns after device execution completes.
    /// </summary>
    public void LinearFloat16(AscendTensor input, AscendTensor weight, AscendTensor output)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentNullException.ThrowIfNull(output);
        lock (_gate)
        {
            Validate(input);
            Validate(weight);
            Validate(output);
            RequireType(input, DotLLM.Core.Tensors.DType.Float16);
            RequireType(weight, DotLLM.Core.Tensors.DType.Float16);
            RequireType(output, DotLLM.Core.Tensors.DType.Float16);
            var x = input.StorageShape;
            var w = weight.StorageShape;
            var y = output.StorageShape;
            if (x.Rank != 2 || w.Rank != 2 || y.Rank != 2 ||
                x[1] != w[1] || y[0] != x[0] || y[1] != w[0])
                throw new ArgumentException($"Linear expects X[M,K], W[N,K], Y[M,N]; got {x}, {w}, {y}.");
            if (ReferenceEquals(output, input) || ReferenceEquals(output, weight))
                throw new ArgumentException("Linear output cannot alias its inputs.");
            MakeCurrent();
            int status = AscendNative.LinearF16(input.DataPointer, weight.DataPointer, output.DataPointer,
                x[0], w[0], x[1], _stream);
            if (status != 0)
            {
                _faulted = true;
                throw new AscendException($"LinearF16(device={DeviceId}, X={x}, W={w})",
                    status, Marshal.PtrToStringUTF8(AscendNative.LastError()));
            }
        }
    }
}
