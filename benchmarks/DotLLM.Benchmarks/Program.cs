using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using DotLLM.Benchmarks.Columns;

var config = ManualConfig.Create(DefaultConfig.Instance)
    .AddColumn(new PrefillTokPerSecColumn())
    .AddColumn(new DecodeTokPerSecColumn());

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);
