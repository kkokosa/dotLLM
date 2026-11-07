using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Ascend;

internal sealed class AscendForwardState : IDisposable
{
    private readonly List<AscendTensor> _storage = [];
    internal AscendTensor Hidden = null!, Residual = null!, Norm = null!, Projection = null!,
        Q = null!, QRot = null!, K = null!, KRot = null!, V = null!, Attention = null!,
        Gate = null!, Up = null!, Activated = null!, Cos = null!, Sin = null!,
        Last = null!, LastNorm = null!, Logits = null!, BiasH = null!, BiasKv = null!, BiasF = null!, CacheK = null!, CacheV = null!;
    internal UnmanagedTensor HostLogits = null!, HostCos = null!, HostSin = null!;
    internal int Rows { get; }
    internal long Bytes => _storage.Sum(t => t.ByteCount) + HostLogits.ByteCount +
        HostCos.ByteCount + HostSin.ByteCount;

    internal AscendForwardState(AscendSession session, ModelConfig c, int rows)
    {
        Rows = rows;
        try
        {
            int h = c.HiddenSize, kv = checked(c.NumKvHeads * c.HeadDim), f = c.IntermediateSize;
            Hidden = Matrix(h); Residual = Matrix(h); Norm = Matrix(h); Projection = Matrix(h);
            Q = Matrix(h); QRot = Matrix(h); K = Matrix(kv); KRot = Matrix(kv); V = Matrix(kv);
            Attention = Matrix(h); Gate = Matrix(f); Up = Matrix(f); Activated = Matrix(f);
            // Bias projections use their own matching-width destination, then copy back.
            BiasH = Matrix(h); BiasKv = Matrix(kv); BiasF = Matrix(f);
            CacheK = AllocateHalf(new(rows, kv)); CacheV = AllocateHalf(new(rows, kv));
            Cos = AllocateHalf(new(rows, c.HeadDim)); Sin = AllocateHalf(new(rows, c.HeadDim));
            Last = Allocate(new(1, h)); LastNorm = Allocate(new(1, h));
            Logits = Allocate(new(1, c.VocabSize));
            HostLogits = UnmanagedTensor.Allocate(new(1, c.VocabSize), DType.Float32);
            HostCos = UnmanagedTensor.Allocate(new(rows, c.HeadDim), DType.Float16);
            HostSin = UnmanagedTensor.Allocate(new(rows, c.HeadDim), DType.Float16);
        }
        catch { Dispose(); throw; }
        return;
        AscendTensor Matrix(int width) => Allocate(new(rows, width));
        AscendTensor AllocateHalf(TensorShape shape)
        {
            var t = session.AllocateFloat16(shape); _storage.Add(t); return t;
        }
        AscendTensor Allocate(TensorShape shape)
        {
            var t = session.AllocateFloat32(shape); _storage.Add(t); return t;
        }
    }

    public void Dispose()
    {
        foreach (var t in _storage) t.Dispose();
        _storage.Clear();
        HostLogits?.Dispose(); HostCos?.Dispose(); HostSin?.Dispose();
    }
}
