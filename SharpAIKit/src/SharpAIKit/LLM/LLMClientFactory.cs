using Microsoft.Extensions.Http;
using Polly;
using Polly.Extensions.Http;
using SharpAIKit.Common;

namespace SharpAIKit.LLM;

/// <summary>
/// Factory for creating LLM client instances.
/// Core principle: Most LLM APIs are OpenAI-compatible, just change URL and API key!
/// </summary>
public static class LLMClientFactory
{
    /// <summary>
    /// Creates an LLM client based on the provider type.
    /// </summary>
    /// <summary>
    /// Creates an LLM client based on the provider type.
    /// </summary>
    public static ILLMClient Create(LLMProviderType providerType, LLMOptions options, Microsoft.Extensions.Logging.ILogger? logger = null)
    {
        var httpClient = CreateHttpClientWithRetry(options);

        return providerType switch
        {
            LLMProviderType.OpenAI => new OpenAIClient(httpClient, options, logger),
            LLMProviderType.Ollama => new OllamaClient(httpClient, options, logger),
            _ => throw new ArgumentOutOfRangeException(nameof(providerType))
        };
    }

    #region Core Method - Support Any OpenAI-Compatible API

    /// <summary>
    /// Creates a client for any OpenAI-compatible API.
    /// Just provide your API key, base URL, and model name!
    /// </summary>
    /// <param name="apiKey">Your API key.</param>
    /// <param name="baseUrl">The API base URL (e.g., https://api.openai.com/v1).</param>
    /// <param name="model">The model name.</param>
    /// <example>
    /// var openai = LLMClientFactory.Create("sk-xxx", "https://api.openai.com/v1", "gpt-4o");
    /// var deepseek = LLMClientFactory.Create("sk-xxx", "https://api.deepseek.com/v1", "deepseek-chat");
    /// var qwen = LLMClientFactory.Create("sk-xxx", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-turbo");
    /// </example>
    public static ILLMClient Create(string apiKey, string baseUrl, string model, Microsoft.Extensions.Logging.ILogger? logger = null)
    {
        var options = new LLMOptions
        {
            ApiKey = apiKey,
            BaseUrl = baseUrl,
            Model = model
        };
        return Create(LLMProviderType.OpenAI, options, logger);
    }

    #endregion

    #region Preset Shortcuts (Optional convenience methods)

    /// <summary>OpenAI (GPT-4o, o1, o3, etc.)</summary>
    public static ILLMClient CreateOpenAI(string apiKey, string model = "gpt-4o")
        => Create(apiKey, "https://api.openai.com/v1", model);

    /// <summary>DeepSeek (DeepSeek-V3, R1, Coder)</summary>
    public static ILLMClient CreateDeepSeek(string apiKey, string model = "deepseek-chat")
        => Create(apiKey, "https://api.deepseek.com/v1", model);

    /// <summary>Qwen/Tongyi</summary>
    public static ILLMClient CreateQwen(string apiKey, string model = "qwen-turbo")
        => Create(apiKey, "https://dashscope.aliyuncs.com/compatible-mode/v1", model);

    /// <summary>Mistral AI</summary>
    public static ILLMClient CreateMistral(string apiKey, string model = "mistral-large-latest")
        => Create(apiKey, "https://api.mistral.ai/v1", model);

    /// <summary>Yi (01.AI)</summary>
    public static ILLMClient CreateYi(string apiKey, string model = "yi-large")
        => Create(apiKey, "https://api.lingyiwanwu.com/v1", model);

    /// <summary>Groq (Ultra-fast Inference)</summary>
    public static ILLMClient CreateGroq(string apiKey, string model = "llama-3.3-70b-versatile")
        => Create(apiKey, "https://api.groq.com/openai/v1", model);

    /// <summary>Moonshot/Kimi</summary>
    public static ILLMClient CreateMoonshot(string apiKey, string model = "moonshot-v1-8k")
        => Create(apiKey, "https://api.moonshot.cn/v1", model);

    /// <summary>Zhipu GLM</summary>
    public static ILLMClient CreateZhipu(string apiKey, string model = "glm-4")
        => Create(apiKey, "https://open.bigmodel.cn/api/paas/v4", model);

    /// <summary>Baichuan</summary>
    public static ILLMClient CreateBaichuan(string apiKey, string model = "Baichuan2-Turbo")
        => Create(apiKey, "https://api.baichuan-ai.com/v1", model);

    /// <summary>Together AI</summary>
    public static ILLMClient CreateTogether(string apiKey, string model = "meta-llama/Llama-3-70b-chat-hf")
        => Create(apiKey, "https://api.together.xyz/v1", model);

    /// <summary>LM Studio (Local)</summary>
    public static ILLMClient CreateLMStudio(string model = "local-model", string baseUrl = "http://localhost:1234/v1")
        => Create("", baseUrl, model);

    #endregion

    #region Ollama (Local)

    /// <summary>
    /// Creates an Ollama client for local models.
    /// Supports: Llama, Qwen, Phi, Mistral, Gemma, DeepSeek, Yi, etc.
    /// </summary>
    public static ILLMClient CreateOllama(string model = "llama3.2", string baseUrl = "http://localhost:11434")
    {
        var options = new LLMOptions
        {
            Model = model,
            BaseUrl = baseUrl
        };
        return Create(LLMProviderType.Ollama, options);
    }

    #endregion

    private static HttpClient CreateHttpClientWithRetry(LLMOptions options)
    {
        var retryPolicy = HttpPolicyExtensions
            .HandleTransientHttpError()
            .WaitAndRetryAsync(options.MaxRetries, retryAttempt =>
                TimeSpan.FromSeconds(Math.Pow(2, retryAttempt)));

        var handler = new PolicyHttpMessageHandler(retryPolicy)
        {
            InnerHandler = new HttpClientHandler()
        };

        return new HttpClient(handler)
        {
            Timeout = TimeSpan.FromSeconds(options.TimeoutSeconds)
        };
    }
}
