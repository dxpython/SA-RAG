namespace SharpAIKit.Agent;

/// <summary>
/// File operation tool providing file read/write capabilities.
/// </summary>
public class FileWriterTool : ToolBase
{
    /// <summary>
    /// Gets or sets the base directory for file operations (security restriction).
    /// </summary>
    public string? BaseDirectory { get; set; }

    /// <summary>
    /// Writes content to a file.
    /// </summary>
    /// <param name="filePath">The file path.</param>
    /// <param name="content">The content to write.</param>
    /// <returns>The operation result.</returns>
    [Tool("write_file", "Writes content to a specified file")]
    public async Task<string> WriteFileAsync(
        [Parameter("The file path")] string filePath,
        [Parameter("The content to write")] string content)
    {
        try
        {
            var safePath = GetSafePath(filePath);
            var directory = Path.GetDirectoryName(safePath);

            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            await File.WriteAllTextAsync(safePath, content);
            return $"File written successfully: {safePath}";
        }
        catch (Exception ex)
        {
            return $"Failed to write file: {ex.Message}";
        }
    }

    /// <summary>
    /// Reads content from a file.
    /// </summary>
    /// <param name="filePath">The file path.</param>
    /// <returns>The file content.</returns>
    [Tool("read_file", "Reads the content of a specified file")]
    public async Task<string> ReadFileAsync(
        [Parameter("The file path")] string filePath)
    {
        try
        {
            var safePath = GetSafePath(filePath);

            if (!File.Exists(safePath))
            {
                return $"File not found: {safePath}";
            }

            var content = await File.ReadAllTextAsync(safePath);
            return $"File Content:\n{content}";
        }
        catch (Exception ex)
        {
            return $"Failed to read file: {ex.Message}";
        }
    }

    /// <summary>
    /// Appends content to a file.
    /// </summary>
    /// <param name="filePath">The file path.</param>
    /// <param name="content">The content to append.</param>
    /// <returns>The operation result.</returns>
    [Tool("append_file", "Appends content to a specified file")]
    public async Task<string> AppendFileAsync(
        [Parameter("The file path")] string filePath,
        [Parameter("The content to append")] string content)
    {
        try
        {
            var safePath = GetSafePath(filePath);
            await File.AppendAllTextAsync(safePath, content);
            return $"Content appended successfully: {safePath}";
        }
        catch (Exception ex)
        {
            return $"Failed to append content: {ex.Message}";
        }
    }

    /// <summary>
    /// Lists the contents of a directory.
    /// </summary>
    /// <param name="directoryPath">The directory path.</param>
    /// <returns>The directory contents.</returns>
    [Tool("list_directory", "Lists the files and subdirectories in a specified directory")]
    public string ListDirectory(
        [Parameter("The directory path")] string directoryPath)
    {
        try
        {
            var safePath = GetSafePath(directoryPath);

            if (!Directory.Exists(safePath))
            {
                return $"Directory not found: {safePath}";
            }

            var entries = Directory.GetFileSystemEntries(safePath)
                .Select(e => Path.GetFileName(e))
                .OrderBy(e => e);

            return $"Directory Contents:\n{string.Join("\n", entries)}";
        }
        catch (Exception ex)
        {
            return $"Failed to list directory: {ex.Message}";
        }
    }

    /// <summary>
    /// Gets a safe path within the base directory.
    /// </summary>
    private string GetSafePath(string path)
    {
        if (string.IsNullOrEmpty(BaseDirectory))
        {
            return Path.GetFullPath(path);
        }

        var fullPath = Path.GetFullPath(Path.Combine(BaseDirectory, path));
        var baseFullPath = Path.GetFullPath(BaseDirectory);

        if (!fullPath.StartsWith(baseFullPath))
        {
            throw new UnauthorizedAccessException("Access to paths outside the base directory is not allowed");
        }

        return fullPath;
    }
}

