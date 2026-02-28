namespace DotLLM.Core.Tensors;

/// <summary>
/// Describes the shape of a tensor: its dimensions, rank, and total element count.
/// Dimensions are stored as a managed array — this is metadata, not tensor data.
/// </summary>
public readonly record struct TensorShape
{
    private readonly int[]? _dimensions;

    /// <summary>Size of each dimension (e.g., [batch, seq_len, hidden_size]).</summary>
    public int[] Dimensions => _dimensions ?? [];

    /// <summary>
    /// Creates a new tensor shape with the specified dimensions.
    /// </summary>
    /// <param name="dimensions">Size of each dimension. Empty for scalar (rank-0) tensors.</param>
    public TensorShape(params ReadOnlySpan<int> dimensions)
    {
        _dimensions = dimensions.ToArray();
    }

    /// <summary>Number of dimensions (axes).</summary>
    public int Rank => Dimensions.Length;

    /// <summary>Total number of elements across all dimensions.</summary>
    public long ElementCount
    {
        get
        {
            long count = 1;
            for (int i = 0; i < Dimensions.Length; i++)
            {
                count *= Dimensions[i];
            }
            return count;
        }
    }

    /// <summary>Gets the size of a specific dimension.</summary>
    /// <param name="axis">Zero-based dimension index.</param>
    public int this[int axis] => Dimensions[axis];

    /// <inheritdoc/>
    public override string ToString() => $"[{string.Join(", ", Dimensions)}]";

    /// <summary>Checks structural equality of dimensions.</summary>
    public bool Equals(TensorShape other) =>
        Dimensions.AsSpan().SequenceEqual(other.Dimensions);

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (int dim in Dimensions)
        {
            hash.Add(dim);
        }
        return hash.ToHashCode();
    }
}
