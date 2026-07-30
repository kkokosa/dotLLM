using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using DotLLM.Benchmarks.Columns;

if (args.Length > 0 && args[0].Equals("--trie-compare", StringComparison.OrdinalIgnoreCase))
{
    DotLLM.Benchmarks.TrieComparisonRunner.Run(args.Length > 1 ? args[1] : null);
    return;
}

var config = ManualConfig.Create(DefaultConfig.Instance)
    .AddColumn(new PrefillTokPerSecColumn())
    .AddColumn(new DecodeTokPerSecColumn());

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);
