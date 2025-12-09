using SharpAIKit.Common;

namespace SharpAIKit.Memory;

/// <summary>
/// Interface for conversation memory systems.
/// Memory stores and retrieves conversation history for context management.
/// </summary>
public interface IMemory
{
    /// <summary>
    /// Adds a message to memory.
    /// </summary>
    Task AddMessageAsync(ChatMessage message);

    /// <summary>
    /// Adds a user-assistant exchange to memory.
    /// </summary>
    Task AddExchangeAsync(string userMessage, string assistantMessage);

    /// <summary>
    /// Gets the relevant messages for a query.
    /// </summary>
    Task<List<ChatMessage>> GetMessagesAsync(string? query = null);

    /// <summary>
    /// Gets the memory as a formatted string for prompt injection.
    /// </summary>
    Task<string> GetContextStringAsync(string? query = null);

    /// <summary>
    /// Clears all messages from memory.
    /// </summary>
    Task ClearAsync();

    /// <summary>
    /// Gets the total number of messages in memory.
    /// </summary>
    Task<int> GetCountAsync();
}

/// <summary>
/// Memory variable that can be injected into prompts.
/// </summary>
public class MemoryVariable
{
    /// <summary>
    /// Gets or sets the variable name.
    /// </summary>
    public string Name { get; set; } = "history";

    /// <summary>
    /// Gets or sets the memory instance.
    /// </summary>
    public IMemory Memory { get; set; } = null!;
}

