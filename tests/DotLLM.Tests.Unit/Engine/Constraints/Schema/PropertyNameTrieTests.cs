using DotLLM.Engine.Constraints.Schema;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints.Schema;

public class PropertyNameTrieTests
{
    [Fact]
    public void Constructor_BuildsTrieFromNames()
    {
        var trie = new PropertyNameTrie(["name", "age"]);
        Assert.True(trie.NodeCount > 1);
    }

    [Fact]
    public void TryGetChild_ValidChar_ReturnsTrue()
    {
        var trie = new PropertyNameTrie(["name"]);
        Assert.True(trie.TryGetChild(0, 'n', out int n));
        Assert.True(trie.TryGetChild(n, 'a', out int na));
        Assert.True(trie.TryGetChild(na, 'm', out int nam));
        Assert.True(trie.TryGetChild(nam, 'e', out _));
    }

    [Fact]
    public void TryGetChild_InvalidChar_ReturnsFalse()
    {
        var trie = new PropertyNameTrie(["name"]);
        Assert.False(trie.TryGetChild(0, 'x', out _));
    }

    [Fact]
    public void IsTerminal_AtCompleteName_ReturnsTrue()
    {
        var trie = new PropertyNameTrie(["name"]);
        trie.TryGetChild(0, 'n', out int n);
        trie.TryGetChild(n, 'a', out int na);
        trie.TryGetChild(na, 'm', out int nam);
        trie.TryGetChild(nam, 'e', out int name);

        Assert.True(trie.IsTerminal(name));
        Assert.Equal("name", trie.GetCompleteName(name));
    }

    [Fact]
    public void IsTerminal_AtPartialName_ReturnsFalse()
    {
        var trie = new PropertyNameTrie(["name"]);
        trie.TryGetChild(0, 'n', out int n);
        trie.TryGetChild(n, 'a', out int na);

        Assert.False(trie.IsTerminal(na));
    }

    [Fact]
    public void SharedPrefixes_BothNamesReachable()
    {
        var trie = new PropertyNameTrie(["name", "namespace"]);

        // Walk to "name"
        trie.TryGetChild(0, 'n', out int n);
        trie.TryGetChild(n, 'a', out int na);
        trie.TryGetChild(na, 'm', out int nam);
        trie.TryGetChild(nam, 'e', out int name);
        Assert.True(trie.IsTerminal(name));

        // Continue to "namespace"
        Assert.True(trie.HasChildren(name)); // 'name' has children ('s')
        trie.TryGetChild(name, 's', out int names);
        trie.TryGetChild(names, 'p', out int namesp);
        trie.TryGetChild(namesp, 'a', out int namespa);
        trie.TryGetChild(namespa, 'c', out int namespac);
        trie.TryGetChild(namespac, 'e', out int namespaceNode);
        Assert.True(trie.IsTerminal(namespaceNode));
        Assert.Equal("namespace", trie.GetCompleteName(namespaceNode));
    }

    [Fact]
    public void GetValidChars_AtRoot_ReturnsFirstChars()
    {
        var trie = new PropertyNameTrie(["alpha", "beta"]);
        var chars = trie.GetValidChars(0).ToList();

        Assert.Contains('a', chars);
        Assert.Contains('b', chars);
        Assert.DoesNotContain('x', chars);
    }

    [Fact]
    public void SingleProperty_Works()
    {
        var trie = new PropertyNameTrie(["x"]);
        Assert.True(trie.TryGetChild(0, 'x', out int x));
        Assert.True(trie.IsTerminal(x));
        Assert.False(trie.HasChildren(x));
    }

    [Fact]
    public void HasChildren_AtRoot_ReturnsTrue()
    {
        var trie = new PropertyNameTrie(["abc"]);
        Assert.True(trie.HasChildren(0));
    }

    [Fact]
    public void HasChildren_AtLeaf_ReturnsFalse()
    {
        var trie = new PropertyNameTrie(["a"]);
        trie.TryGetChild(0, 'a', out int a);
        Assert.False(trie.HasChildren(a));
    }
}
