namespace SharpAIKit.MultiModal;

/// <summary>
/// Represents a content part in a multimodal message.
/// </summary>
public abstract class ContentPart
{
    /// <summary>
    /// Gets the content type.
    /// </summary>
    public abstract string Type { get; }

    /// <summary>
    /// Converts to the API representation.
    /// </summary>
    public abstract object ToApiRepresentation();
}

/// <summary>
/// Text content part.
/// </summary>
public class TextContent : ContentPart
{
    /// <inheritdoc/>
    public override string Type => "text";

    /// <summary>
    /// Gets or sets the text content.
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Creates a new text content.
    /// </summary>
    public TextContent(string text)
    {
        Text = text;
    }

    /// <inheritdoc/>
    public override object ToApiRepresentation() => new
    {
        type = "text",
        text = Text
    };
}

/// <summary>
/// Image content part.
/// </summary>
public class ImageContent : ContentPart
{
    /// <inheritdoc/>
    public override string Type => "image_url";

    /// <summary>
    /// Gets or sets the image URL or base64 data.
    /// </summary>
    public string Url { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the detail level (auto, low, high).
    /// </summary>
    public string Detail { get; set; } = "auto";

    /// <summary>
    /// Creates image content from a URL.
    /// </summary>
    public static ImageContent FromUrl(string url, string detail = "auto") => new()
    {
        Url = url,
        Detail = detail
    };

    /// <summary>
    /// Creates image content from a local file.
    /// </summary>
    public static ImageContent FromFile(string filePath, string detail = "auto")
    {
        var bytes = File.ReadAllBytes(filePath);
        var base64 = Convert.ToBase64String(bytes);
        var mimeType = GetMimeType(filePath);
        
        return new ImageContent
        {
            Url = $"data:{mimeType};base64,{base64}",
            Detail = detail
        };
    }

    /// <summary>
    /// Creates image content from bytes.
    /// </summary>
    public static ImageContent FromBytes(byte[] data, string mimeType = "image/png", string detail = "auto")
    {
        var base64 = Convert.ToBase64String(data);
        return new ImageContent
        {
            Url = $"data:{mimeType};base64,{base64}",
            Detail = detail
        };
    }

    /// <inheritdoc/>
    public override object ToApiRepresentation() => new
    {
        type = "image_url",
        image_url = new
        {
            url = Url,
            detail = Detail
        }
    };

    private static string GetMimeType(string filePath)
    {
        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        return extension switch
        {
            ".jpg" or ".jpeg" => "image/jpeg",
            ".png" => "image/png",
            ".gif" => "image/gif",
            ".webp" => "image/webp",
            ".bmp" => "image/bmp",
            _ => "application/octet-stream"
        };
    }
}

/// <summary>
/// A multimodal chat message that can contain text and images.
/// </summary>
public class MultiModalMessage
{
    /// <summary>
    /// Gets or sets the role.
    /// </summary>
    public string Role { get; set; } = "user";

    /// <summary>
    /// Gets or sets the content parts.
    /// </summary>
    public List<ContentPart> Content { get; set; } = new();

    /// <summary>
    /// Creates a user message.
    /// </summary>
    public static MultiModalMessage User(params ContentPart[] parts) => new()
    {
        Role = "user",
        Content = parts.ToList()
    };

    /// <summary>
    /// Creates a user message with text and images.
    /// </summary>
    public static MultiModalMessage User(string text, params string[] imageUrls)
    {
        var content = new List<ContentPart> { new TextContent(text) };
        content.AddRange(imageUrls.Select(url => ImageContent.FromUrl(url)));
        return new MultiModalMessage { Role = "user", Content = content };
    }

    /// <summary>
    /// Adds text content.
    /// </summary>
    public MultiModalMessage AddText(string text)
    {
        Content.Add(new TextContent(text));
        return this;
    }

    /// <summary>
    /// Adds image content from URL.
    /// </summary>
    public MultiModalMessage AddImageUrl(string url, string detail = "auto")
    {
        Content.Add(ImageContent.FromUrl(url, detail));
        return this;
    }

    /// <summary>
    /// Adds image content from file.
    /// </summary>
    public MultiModalMessage AddImageFile(string filePath, string detail = "auto")
    {
        Content.Add(ImageContent.FromFile(filePath, detail));
        return this;
    }

    /// <summary>
    /// Converts to API representation.
    /// </summary>
    public object ToApiRepresentation() => new
    {
        role = Role,
        content = Content.Select(c => c.ToApiRepresentation()).ToList()
    };
}

/// <summary>
/// Builder for creating multimodal messages fluently.
/// </summary>
public class MultiModalMessageBuilder
{
    private string _role = "user";
    private readonly List<ContentPart> _parts = new();

    /// <summary>
    /// Sets the role.
    /// </summary>
    public MultiModalMessageBuilder WithRole(string role)
    {
        _role = role;
        return this;
    }

    /// <summary>
    /// Adds text.
    /// </summary>
    public MultiModalMessageBuilder AddText(string text)
    {
        _parts.Add(new TextContent(text));
        return this;
    }

    /// <summary>
    /// Adds an image from URL.
    /// </summary>
    public MultiModalMessageBuilder AddImage(string url, string detail = "auto")
    {
        _parts.Add(ImageContent.FromUrl(url, detail));
        return this;
    }

    /// <summary>
    /// Adds an image from file.
    /// </summary>
    public MultiModalMessageBuilder AddImageFile(string filePath, string detail = "auto")
    {
        _parts.Add(ImageContent.FromFile(filePath, detail));
        return this;
    }

    /// <summary>
    /// Adds an image from bytes.
    /// </summary>
    public MultiModalMessageBuilder AddImageBytes(byte[] data, string mimeType = "image/png", string detail = "auto")
    {
        _parts.Add(ImageContent.FromBytes(data, mimeType, detail));
        return this;
    }

    /// <summary>
    /// Builds the message.
    /// </summary>
    public MultiModalMessage Build() => new()
    {
        Role = _role,
        Content = _parts.ToList()
    };
}

/// <summary>
/// Extension methods for multimodal support.
/// </summary>
public static class MultiModalExtensions
{
    /// <summary>
    /// Creates a multimodal message builder.
    /// </summary>
    public static MultiModalMessageBuilder CreateMultiModalMessage() => new();

    /// <summary>
    /// Creates a simple vision message with text and one image.
    /// </summary>
    public static MultiModalMessage CreateVisionMessage(string text, string imageUrl)
    {
        return MultiModalMessage.User(text, imageUrl);
    }

    /// <summary>
    /// Creates a vision message from a local image file.
    /// </summary>
    public static MultiModalMessage CreateVisionMessageFromFile(string text, string imagePath)
    {
        return new MultiModalMessage()
            .AddText(text)
            .AddImageFile(imagePath);
    }
}

