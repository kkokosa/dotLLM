namespace DotLLM.Ascend;

/// <summary>An AscendCL or ACLNN call failed.</summary>
public sealed class AscendException : Exception
{
    /// <summary>The failed native operation.</summary>
    public string Operation { get; }

    /// <summary>The native status code.</summary>
    public int Status { get; }

    /// <summary>Creates an exception with the native diagnostic text.</summary>
    public AscendException(string operation, int status, string? detail)
        : base($"{operation} failed (status={status}): {detail ?? "no native diagnostic"}")
    {
        Operation = operation;
        Status = status;
    }
}
