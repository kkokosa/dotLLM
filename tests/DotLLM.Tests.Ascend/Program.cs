using System.Runtime.InteropServices;
using System.Text.Json;
using DotLLM.Ascend;
using DotLLM.Core.Tensors;

try
{
    ReferenceSelfTest();
    if (args is ["--reference-self-test"])
    {
        Console.WriteLine("""{"status":"PASS","scope":"CPU reference self-test only; no NPU executed"}""");
        return 0;
    }
    if (args is ["--model-test", var modelPath])
    {
        ModelSmoke.Run(modelPath, null);
        return 0;
    }
    if (args is ["--model-test", var inputPath, "--reference", var referencePath])
    {
        ModelSmoke.Run(inputPath, referencePath);
        return 0;
    }
    if (args is ["--model-test", var mathInput, "--reference", var cpuInput, "--math-reference", var mathInputPath])
    {
        ModelSmoke.Run(mathInput, cpuInput, mathInputPath);
        return 0;
    }
    if (args is ["--emit-text", var emitModel, var emitPath])
    {
        TextSmoke.Emit(emitModel, emitPath);
        return 0;
    }
    if (args is ["--text-test", var textModel, var casesPath, var goldenPath])
    {
        TextSmoke.Run(textModel, casesPath, goldenPath);
        return 0;
    }
    if (args is ["--capability-self-test", var capabilityPath])
    {
        using var file = DotLLM.Models.Gguf.GgufFile.Open(capabilityPath);
        var c = DotLLM.Models.Gguf.GgufModelConfigExtractor.Extract(file.Metadata);
        Expect<InvalidDataException>(() => AscendTransformerModel.LoadFromGguf(file, c with { HiddenSize = 0 }));
        Expect<InvalidDataException>(() => AscendTransformerModel.LoadFromGguf(file, c with { VocabSize = c.VocabSize + 1 }));
        Expect<NotSupportedException>(() => AscendTransformerModel.LoadFromGguf(file, c with { SlidingWindowSize = 16 }));
        Expect<NotSupportedException>(() => AscendTransformerModel.LoadFromGguf(file, c with { NormType = DotLLM.Core.Configuration.NormType.LayerNorm }));
        Expect<NotSupportedException>(() => AscendTransformerModel.LoadFromGguf(file, c with { RoPEConfig = c.RoPEConfig!.Value with { ScalingFactor = 2 } }));
        Expect<NotSupportedException>(() => AscendTransformerModel.LoadFromGguf(file, c with { RoPEConfig = c.RoPEConfig!.Value with { Type = (DotLLM.Core.Configuration.RoPEType)99 } }));
        Console.WriteLine("""{"test":"capability_rejection","status":"PASS","scope":"host validation before native initialization","cases":6}""");
        return 0;
    }
    if (args is ["--fault-test"])
    {
        using var session = AscendSession.Create();
        using var input = session.AllocateFloat16(new(1, 2));
        using var weight = session.AllocateFloat16(new(1, 2));
        using var output = session.AllocateFloat16(new(1, 1));
        Expect<AscendException>(() => session.LinearFloat16(input, weight, output));
        Expect<InvalidOperationException>(() => session.LinearFloat16(input, weight, output));
        session.Dispose();
        if (input.DataPointer != 0 || weight.DataPointer != 0 || output.DataPointer != 0)
            throw new Exception("Faulted session did not clean all buffers.");
        Console.WriteLine("""{"test":"injected_bridge_failure_cleanup","status":"PASS","scope":"synthetic bridge error with real AscendCL allocations"}""");
        return 0;
    }
    if (args is ["--memory-test", var memoryPath])
    {
        using var observer = AscendSession.Create();
        using var file = DotLLM.Models.Gguf.GgufFile.Open(memoryPath);
        var c = DotLLM.Models.Gguf.GgufModelConfigExtractor.Extract(file.Metadata);
        // Warm ACLNN's internal caches before measuring owner allocation return.
        using (var warm = AscendTransformerModel.LoadFromGguf(file, c))
            using (warm.Forward([1, 2], [0, 1], -1)) { }
        long baseline = observer.GetMemoryInfo().FreeBytes;
        var after = new List<long>();
        for (int repeat = 0; repeat < 3; repeat++)
        {
            using (var m = AscendTransformerModel.LoadFromGguf(file, c))
            using (var cache = m.CreateKvCache(16))
                using (m.Forward([1, 2], [0, 1], -1, cache)) { }
            after.Add(observer.GetMemoryInfo().FreeBytes);
            if (baseline - after[^1] > 64 * 1024 * 1024)
                throw new Exception("Device allocations did not return within the fixed 64 MiB runtime-cache budget.");
        }
        Console.WriteLine(JsonSerializer.Serialize(new { test = "model_unload_memory", status = "PASS", baselineFreeBytes = baseline,
            afterFreeBytes = after, allowedRetainedBytes = 64 * 1024 * 1024 }));
        return 0;
    }
    if (args is ["--nonfinite-test", var invalidWeightPath])
    {
        using var observer = AscendSession.Create();
        using var file = DotLLM.Models.Gguf.GgufFile.Open(invalidWeightPath);
        var c = DotLLM.Models.Gguf.GgufModelConfigExtractor.Extract(file.Metadata);
        long baseline = observer.GetMemoryInfo().FreeBytes;
        Expect<InvalidDataException>(() => AscendTransformerModel.LoadFromGguf(file, c));
        long after = observer.GetMemoryInfo().FreeBytes;
        if (baseline - after > 64 * 1024 * 1024) throw new Exception("Failed load leaked device allocations.");
        Console.WriteLine(JsonSerializer.Serialize(new { test = "nonfinite_weights_cleanup", status = "PASS", baselineFreeBytes = baseline,
            afterFreeBytes = after, allowedRetainedBytes = 64 * 1024 * 1024 }));
        return 0;
    }
    int deviceId = args.Length == 0 ? 0 :
        args is ["--device", var value] && int.TryParse(value, out int parsed) && parsed >= 0
            ? parsed : throw new ArgumentException("Usage: [--device N] | --reference-self-test");
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        phase = "environment", deviceId, runtime = RuntimeInformation.FrameworkDescription,
        architecture = RuntimeInformation.ProcessArchitecture.ToString(),
        os = RuntimeInformation.OSDescription
    }));

    // A second session must survive disposal of the first, even on the same device.
    using (var first = AscendSession.Create(deviceId))
    using (var second = AscendSession.Create(deviceId))
    {
        RunCase(first, 3, 5, 7);
        KernelSmoke.Run(first);
        ValidateContracts(first, second);
        ValidateBackend(deviceId);
        first.Dispose();
        RunCase(second, 1, 19, 37);
        Task.Run(() => RunCase(second, 5, 33, 65)).GetAwaiter().GetResult();
        RunCase(second, 16, 32, 64);
        RunCase(second, 1, 576, 576);
        RunCase(second, 7, 129, 1025);
    }
    // Reinitialization after the last owner exits is part of the runtime contract.
    using (var recreated = AscendSession.Create(deviceId))
        RunCase(recreated, 2, 11, 13);
    Console.WriteLine("""{"status":"PASS","scope":"Ascend FP16 kernels and runtime smoke","linearCases":7,"iterationsPerCase":3}""");
    return 0;
}
catch (Exception error)
{
    Console.Error.WriteLine(JsonSerializer.Serialize(new { status = "FAIL", error = error.ToString() }));
    return 1;
}

static unsafe void RunCase(AscendSession session, int m, int n, int k)
{
    using var hostX = UnmanagedTensor.Allocate(new(m, k), DType.Float16);
    using var hostW = UnmanagedTensor.Allocate(new(n, k), DType.Float16);
    using var hostY = UnmanagedTensor.Allocate(new(m, n), DType.Float16);
    using var hostReference = UnmanagedTensor.Allocate(new(m, n), DType.Float32);
    using var dx = session.AllocateFloat16(new(m, k));
    using var dw = session.AllocateFloat16(new(n, k));
    using var dy = session.AllocateFloat16(new(m, n));
    var x = new Span<Half>((void*)hostX.DataPointer, m * k);
    var w = new Span<Half>((void*)hostW.DataPointer, n * k);
    var y = new Span<Half>((void*)hostY.DataPointer, m * n);
    var reference = new Span<float>((void*)hostReference.DataPointer, m * n);
    double maxAbsoluteError = 0, sumSquaredError = 0, sumSquaredReference = 0;
    for (int iteration = 0; iteration < 3; iteration++)
    {
        Fill(x, (uint)(17 + iteration));
        Fill(w, (uint)(91 + iteration));
        ReferenceLinear(x, w, reference, m, n, k);
        session.Upload(dx, x);
        session.Upload(dw, w);
        // Poison output to ensure every element is actually written.
        y.Fill(Half.NaN);
        session.Upload(dy, y);
        session.LinearFloat16(dx, dw, dy);
        session.Download(dy, y);
        for (int i = 0; i < y.Length; i++)
        {
            double expected = reference[i], actual = (float)y[i];
            double error = Math.Abs(actual - expected);
            double tolerance = 0.002 + 0.002 * Math.Abs(expected);
            if (!double.IsFinite(actual) || error > tolerance)
                throw new InvalidOperationException(
                    $"Linear({m},{n},{k}) iteration={iteration} index={i}: expected={expected}, actual={actual}, tolerance={tolerance}.");
            maxAbsoluteError = Math.Max(maxAbsoluteError, error);
            sumSquaredError += error * error;
            sumSquaredReference += expected * expected;
        }
    }
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        test = "linear_f16", m, n, k, status = "PASS", maxAbsoluteError,
        normalizedRmse = Math.Sqrt(sumSquaredError / Math.Max(sumSquaredReference, 1e-30)),
        atol = 0.002, rtol = 0.002, math = "KEEP_DTYPE",
        reference = "scalar double accumulation of the exact uploaded FP16 values"
    }));
}

static unsafe void ValidateContracts(AscendSession first, AscendSession second)
{
    var originalShape = new TensorShape(2, 3);
    using var tensor = first.AllocateFloat16(originalShape);
    originalShape.Dimensions[0] = 99;
    tensor.Shape.Dimensions[1] = 99;
    if (tensor.Shape != new TensorShape(2, 3) || tensor.ElementCount != 6)
        throw new InvalidOperationException("Tensor shape escaped mutable ownership.");
    Span<Half> input = stackalloc Half[] { (Half)0, (Half)1, (Half)(-2), (Half)0.125, (Half)3, (Half)(-0.5) };
    Span<Half> output = stackalloc Half[6];
    first.Upload(tensor, input);
    first.Download(tensor, output);
    if (!MemoryMarshal.AsBytes(input).SequenceEqual(MemoryMarshal.AsBytes(output)))
        throw new InvalidOperationException("H2D/D2H round trip changed FP16 bits.");
    using var foreign = second.AllocateFloat16(new(2, 3));
    using var wrong = first.AllocateFloat16(new(4, 4));
    Expect<ArgumentException>(() => first.LinearFloat16(tensor, foreign, tensor));
    Expect<ArgumentException>(() => first.LinearFloat16(tensor, wrong, tensor));
    using var square = first.AllocateFloat16(new(2, 2));
    Expect<ArgumentException>(() => first.LinearFloat16(square, square, square));
    Expect<ArgumentException>(() => first.Upload(tensor, ReadOnlySpan<Half>.Empty));
    Expect<ArgumentOutOfRangeException>(() => first.AllocateFloat16(new(2, -1)));
    Expect<OverflowException>(() => first.AllocateFloat16(new(int.MaxValue, int.MaxValue, 3)));
    tensor.Dispose();
    tensor.Dispose();
    Expect<ObjectDisposedException>(() => first.Upload(tensor, ReadOnlySpan<Half>.Empty));
    using (var owner = AscendSession.Create(first.DeviceId))
    {
        var orphan = owner.AllocateFloat16(new(1));
        owner.Dispose();
        if (orphan.DataPointer != 0) throw new InvalidOperationException("Session did not release its tensor.");
        orphan.Dispose();
        Expect<ObjectDisposedException>(() => owner.AllocateFloat16(new(1)));
    }
    Console.WriteLine("""{"test":"ownership_and_copy","status":"PASS"}""");
}

static void ReferenceLinear(ReadOnlySpan<Half> x, ReadOnlySpan<Half> w, Span<float> y, int m, int n, int k)
{
    for (int row = 0; row < m; row++)
        for (int column = 0; column < n; column++)
        {
            double sum = 0;
            for (int inner = 0; inner < k; inner++)
                sum += (double)x[row * k + inner] * (double)w[column * k + inner];
            y[row * n + column] = (float)sum;
        }
}

static void ReferenceSelfTest()
{
    ReadOnlySpan<Half> x = [(Half)1, (Half)2, (Half)3, (Half)(-1), (Half)0, (Half)2];
    ReadOnlySpan<Half> w = [(Half)1, (Half)0, (Half)(-1), (Half)2, (Half)1, (Half)0,
        (Half)0, (Half)0, (Half)1, (Half)(-1), (Half)2, (Half)1];
    Span<float> y = stackalloc float[8];
    ReferenceLinear(x, w, y, 2, 4, 3);
    ReadOnlySpan<float> expected = [-2, 4, 3, 6, -3, -2, 2, 3];
    if (!y.SequenceEqual(expected)) throw new InvalidOperationException("Scalar reference self-test failed.");
    Expect<ArgumentOutOfRangeException>(() => AscendSession.Create(-1));
    if (!AscendDevice.TryParse("ASCEND:12", out int ordinal) || ordinal != 12 ||
        AscendDevice.TryParse("cpu", out _) || AscendDevice.TryParse("gpu:0", out _))
        throw new InvalidOperationException("Ascend device routing failed.");
    Expect<ArgumentException>(() => AscendDevice.TryParse("ascend:-1", out _));
    Expect<ArgumentException>(() => AscendDevice.TryParse("ascend:1:2", out _));
    Expect<NotSupportedException>(() => AscendDevice.ValidateOptions("ascend", 0, "f32", "f32"));
    Expect<NotSupportedException>(() => AscendDevice.ValidateOptions("ascend", null, "q8_0", "f32"));
    Expect<NotSupportedException>(() => AscendDevice.ValidateOptions("ascend", null, "f32", "f32", paged: true));
    Expect<NotSupportedException>(() => AscendDevice.ValidateOptions("ascend", null, "f32", "f32", speculativeModel: "draft"));
}

static void Fill(Span<Half> values, uint state)
{
    for (int i = 0; i < values.Length; i++)
    {
        state = unchecked(state * 1664525 + 1013904223);
        values[i] = (Half)(((int)(state >> 16) - 32768) / 65536.0f);
    }
}

static void Expect<T>(Action action) where T : Exception
{
    try { action(); }
    catch (T) { return; }
    throw new InvalidOperationException($"Expected {typeof(T).Name}.");
}

static unsafe void ValidateBackend(int deviceId)
{
    using var backend = new AscendBackend(deviceId);
    if (backend.DeviceCount <= deviceId) throw new Exception("Device count mismatch.");
    using var host = backend.AllocateOnDevice(-1, new(2, 3), DType.Float16);
    using var device = backend.AllocateOnDevice(deviceId, new(2, 3), DType.Float16);
    using var copied = backend.AllocateOnDevice(deviceId, new(2, 3), DType.Float16);
    using var output = backend.AllocateOnDevice(-1, new(2, 3), DType.Float16);
    new Span<Half>((void*)host.DataPointer, 6).Fill((Half)1.25);
    backend.CopyBetweenDevices(host, device);
    backend.CopyBetweenDevices(device, copied);
    backend.CopyBetweenDevices(copied, output);
    if (!new ReadOnlySpan<Half>((void*)host.DataPointer, 6).SequenceEqual(new ReadOnlySpan<Half>((void*)output.DataPointer, 6)))
        throw new Exception("Backend copy differs.");
    Expect<NotSupportedException>(() => backend.AllocateOnDevice(deviceId, new(1), DType.Float32));
    Expect<ArgumentException>(() => backend.AllocateOnDevice(deviceId + 1, new(1), DType.Float16));
    Expect<NotSupportedException>(() => backend.AllReduce([device]));
    Expect<NotSupportedException>(() => backend.Send(device, deviceId));
    Expect<NotSupportedException>(() => backend.Receive(deviceId, new(1), DType.Float16));
    using var other = new AscendBackend(deviceId);
    using var foreign = other.AllocateOnDevice(deviceId, new(2, 3), DType.Float16);
    Expect<ArgumentException>(() => backend.CopyBetweenDevices(foreign, device));
    Console.WriteLine("""{"test":"backend_contract","status":"PASS"}""");
}
