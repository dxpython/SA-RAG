namespace SharpAIKit.Agent;

/// <summary>
/// Web search tool (mock implementation).
/// In production, connect to a real search API (Bing, Google, etc.).
/// </summary>
public class WebSearchTool : ToolBase
{
    /// <summary>
    /// Searches the web for information.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <returns>Search results.</returns>
    [Tool("web_search", "Searches the web for the latest information")]
    public Task<string> SearchAsync(
        [Parameter("The search keywords")] string query)
    {
        // Mock implementation - in production, connect to a real search API
        var results = $"""
            Search Results (Mock):
            1. {query} - Wikipedia
               A detailed introduction about {query}...
            2. {query} Latest News - News Website
               Today's latest updates about {query}...
            3. {query} Tutorial - Tech Blog
               How to learn and use {query}...
            """;

        return Task.FromResult(results);
    }

    /// <summary>
    /// Fetches content from a URL.
    /// </summary>
    /// <param name="url">The URL to fetch.</param>
    /// <returns>The webpage content.</returns>
    [Tool("fetch_url", "Fetches the content of a specified URL")]
    public async Task<string> FetchUrlAsync(
        [Parameter("The URL to access")] string url)
    {
        try
        {
            using var httpClient = new HttpClient();
            httpClient.Timeout = TimeSpan.FromSeconds(10);
            httpClient.DefaultRequestHeaders.Add("User-Agent", "SharpAIKit/1.0");

            var content = await httpClient.GetStringAsync(url);

            // Truncate to first 1000 characters
            if (content.Length > 1000)
            {
                content = content[..1000] + "... (content truncated)";
            }

            return $"Webpage Content:\n{content}";
        }
        catch (Exception ex)
        {
            return $"Failed to fetch webpage: {ex.Message}";
        }
    }
}

