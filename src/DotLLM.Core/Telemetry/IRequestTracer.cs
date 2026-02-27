using System.Diagnostics;

namespace DotLLM.Core.Telemetry;

/// <summary>
/// Creates distributed tracing spans via <c>System.Diagnostics.Activity</c>.
/// Integrates with OpenTelemetry exporters when configured.
/// </summary>
public interface IRequestTracer
{
    /// <summary>
    /// Starts a new tracing span as a child of the current activity.
    /// </summary>
    /// <param name="operationName">Name of the operation (e.g., "prefill", "decode", "sample").</param>
    /// <returns>An <see cref="Activity"/> representing the span, or null if tracing is not active.</returns>
    Activity? StartSpan(string operationName);

    /// <summary>
    /// Starts a top-level request span encompassing the entire inference request.
    /// </summary>
    /// <param name="requestId">Unique identifier for the request.</param>
    /// <returns>An <see cref="Activity"/> representing the request span, or null if tracing is not active.</returns>
    Activity? StartRequestSpan(string requestId);
}
