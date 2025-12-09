namespace SharpAIKit.Common;

/// <summary>
/// Represents a chat message in a conversation.
/// </summary>
public class ChatMessage
{
    /// <summary>
    /// Gets or sets the role of the message sender (system, user, assistant).
    /// </summary>
    public string Role { get; set; } = "user";

    /// <summary>
    /// Gets or sets the content of the message.
    /// </summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// Creates a user message.
    /// </summary>
    /// <param name="content">The message content.</param>
    /// <returns>A new ChatMessage with role "user".</returns>
    public static ChatMessage User(string content) => new() { Role = "user", Content = content };

    /// <summary>
    /// Creates an assistant message.
    /// </summary>
    /// <param name="content">The message content.</param>
    /// <returns>A new ChatMessage with role "assistant".</returns>
    public static ChatMessage Assistant(string content) => new() { Role = "assistant", Content = content };

    /// <summary>
    /// Creates a system message.
    /// </summary>
    /// <param name="content">The message content.</param>
    /// <returns>A new ChatMessage with role "system".</returns>
    public static ChatMessage System(string content) => new() { Role = "system", Content = content };
}

