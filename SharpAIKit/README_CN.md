<div align="center">

## SharpAIKit：统一的 AI/LLM 工具包 for .NET

### 🎯 比 LangChain 更强大，比 LangChain 更简洁

<img src="imgs/logo.jpg" alt="SharpAIKit Logo" width="900">

[![.NET Version](https://img.shields.io/badge/.NET-8.0-512BD4?style=for-the-badge&logo=dotnet&logoColor=white)](https://dotnet.microsoft.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![NuGet](https://img.shields.io/badge/NuGet-v2.0.0-004880?style=for-the-badge&logo=nuget&logoColor=white)](https://www.nuget.org/)

[English](README_EN.md) | [🏠 首页](README.md)

</div>

---

## 🆚 SharpAIKit vs LangChain

| 功能 | SharpAIKit | LangChain |
|:-----|:----------:|:---------:|
| **链式调用 (LCEL)** | ✅ 管道操作符 | ✅ |
| **对话记忆** | ✅ 5种策略 | ✅ |
| **提示模板** | ✅ 类型安全 | ✅ |
| **输出解析** | ✅ 强类型泛型 | ✅ |
| **文档加载器** | ✅ 多格式 | ✅ |
| **可观测性** | ✅ 全链路追踪 | ✅ |
| **多模态** | ✅ 图像支持 | ✅ |
| **高级Agent** | ✅ ReAct/多Agent | ✅ |
| **Code Interpreter** | ✅ **原生 C# (Roslyn)** | ❌ Python 依赖 |
| **图编排** | ✅ **SharpGraph (FSM)** | ⚠️ LangGraph (新) |
| **自动优化** | ✅ **DSPy-style** | ❌ 无 |
| **类型安全** | ✅ **C# 强类型** | ❌ Python 弱类型 |
| **性能** | ✅ **原生编译** | ❌ 解释执行 |
| **代码简洁度** | ✅ **极简 API** | ❌ 抽象层层嵌套 |
| **依赖** | ✅ **极少** | ❌ 大量依赖 |

---

## 📋 目录

- [功能特性](#-功能特性)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [链式调用 (Chain)](#-链式调用-chain)
- [对话记忆 (Memory)](#-对话记忆-memory)
- [提示模板 (Prompt)](#-提示模板-prompt)
- [输出解析 (Output Parser)](#-输出解析-output-parser)
- [文档加载 (Document Loader)](#-文档加载-document-loader)
- [可观测性 (Callback)](#-可观测性-callback)
- [多模态 (MultiModal)](#-多模态-multimodal)
- [高级 Agent](#-高级-agent)
- [RAG 引擎](#-rag-引擎)
- [🔮 Native C# Code Interpreter](#-native-c-code-interpreter) ⭐ **杀手级功能**
- [🕸️ SharpGraph 图编排](#️-sharpgraph-图编排) ⭐ **杀手级功能**
- [🧬 DSPy-style Optimizer](#-dspy-style-optimizer) ⭐ **杀手级功能**
- [支持的提供商](#-支持的提供商)

---

## ✨ 功能特性

| 模块 | 功能 | 说明 |
|:-----|:-----|:-----|
| 🔗 **Chain** | 链式调用 | LCEL 风格管道组合，支持 `\|` 操作符 |
| 🧠 **Memory** | 对话记忆 | Buffer、Window、Summary、Vector、Entity 五种策略 |
| 📝 **Prompt** | 提示模板 | 变量替换、Chat模板、Few-shot学习 |
| 📤 **Output** | 输出解析 | JSON、Boolean、List、XML、Regex 解析器 |
| 📄 **Loader** | 文档加载 | Text、CSV、JSON、Markdown、Web 多格式 |
| 📊 **Callback** | 可观测性 | Console、Logging、Metrics、File 全链路追踪 |
| 🖼️ **MultiModal** | 多模态 | 图像URL、本地文件、Base64 支持 |
| 🤖 **Agent** | 智能代理 | ReAct、Plan-Execute、Multi-Agent 系统 |
| 📚 **RAG** | 检索增强 | 文档索引、向量搜索、智能问答 |
| 🔮 **Code Interpreter** | 代码解释器 | **原生 C# 代码执行**，无需 Python，基于 Roslyn |
| 🕸️ **SharpGraph** | 图编排 | **有限状态机**，支持循环和复杂分支 |
| 🧬 **DSPy Optimizer** | 自动优化 | **自动提示词优化**，越用越聪明 |

---

## 📦 安装

```bash
dotnet add package SharpAIKit
```

---

## 🚀 快速开始

```csharp
using SharpAIKit.LLM;

// 三行代码，支持任意 OpenAI 兼容 API
var client = LLMClientFactory.Create("your-api-key", "https://api.deepseek.com/v1", "deepseek-chat");
var response = await client.ChatAsync("你好！");
Console.WriteLine(response);

// 流式输出
await foreach (var chunk in client.ChatStreamAsync("讲一个故事"))
{
    Console.Write(chunk);
}
```

---

## 🔗 链式调用 (Chain)

**LCEL 风格的管道组合，像 LangChain 一样优雅！**

```csharp
using SharpAIKit.Chain;
using SharpAIKit.Prompt;

// 创建链组件
var prompt = PromptTemplate.FromTemplate("将以下内容翻译成{language}: {input}");
var llm = new LLMChain(client);

// 管道组合 (使用 | 操作符，像 LangChain LCEL!)
var chain = prompt.Pipe(llm);

// 执行
var result = await chain.InvokeAsync(new ChainContext()
    .Set("language", "法语")
    .Set("input", "你好世界"));

Console.WriteLine(result.Output);

// 并行执行多个链
var parallel = ChainExtensions.Parallel(
    new LLMChain(client, "你是乐观主义者"),
    new LLMChain(client, "你是现实主义者"),
    new LLMChain(client, "你是批评家")
);

// 条件分支
var branched = chain.Branch(
    ctx => ctx.Output.Contains("error"),
    trueBranch: errorHandlerChain,
    falseBranch: successChain
);
```

---

## 🧠 对话记忆 (Memory)

**5种记忆策略，比 LangChain 更丰富！**

```csharp
using SharpAIKit.Memory;

// 1. Buffer Memory - 保留最近 N 条消息
var buffer = new BufferMemory { MaxMessages = 20 };

// 2. Window Buffer - 保留最近 N 轮对话
var window = new WindowBufferMemory { WindowSize = 5 };

// 3. Summary Memory - 自动总结旧对话
var summary = new SummaryMemory(client) { RecentMessagesCount = 6 };

// 4. Vector Memory - 语义搜索相关对话
var vector = new VectorMemory(client) { TopK = 5 };

// 5. Entity Memory - 提取和追踪实体
var entity = new EntityMemory(client);

// 使用
await buffer.AddExchangeAsync("什么是 Python?", "Python 是一种编程语言...");
var context = await buffer.GetContextStringAsync();
```

---

## 📝 提示模板 (Prompt)

**类型安全的模板系统！**

```csharp
using SharpAIKit.Prompt;

// 简单模板
var template = PromptTemplate.FromTemplate("你是{role}，回答问题：{input}");
var prompt = template.Format(("role", "AI助手"), ("input", "什么是C#?"));

// 带部分变量
var withPartials = PromptTemplate.FromTemplate("当前时间: {time}\n用户: {input}")
    .WithPartial("time", () => DateTime.Now.ToString());

// Chat 模板
var chatTemplate = new ChatPromptTemplate()
    .AddSystemMessage("你是{role}助手")
    .AddHistoryPlaceholder("history")
    .AddUserMessage("{input}");

// Few-shot 模板
var fewShot = new FewShotPromptTemplate(
    prefix: "对以下文本进行情感分类:",
    suffix: "文本: {input}\n情感:",
    exampleTemplate: "文本: {input}\n情感: {output}"
)
.AddExample("我喜欢这个产品!", "积极")
.AddExample("太糟糕了", "消极");
```

---

## 📤 输出解析 (Output Parser)

**强类型泛型解析，比 LangChain 更安全！**

```csharp
using SharpAIKit.Output;

// JSON 解析为强类型对象
var jsonParser = new JsonOutputParser<ProductReview>();
var review = jsonParser.Parse(llmOutput);
Console.WriteLine(review.Rating);  // 强类型访问!

// Boolean 解析
var boolParser = new BooleanParser();
bool isTrue = boolParser.Parse("yes");  // true

// 列表解析
var listParser = new CommaSeparatedListParser();
List<string> items = listParser.Parse("apple, banana, orange");

// XML 标签解析
var xmlParser = new XMLTagParser("answer", "reasoning");
var result = xmlParser.Parse("<answer>42</answer><reasoning>...</reasoning>");

// 获取格式指令（用于 Prompt）
string instructions = jsonParser.GetFormatInstructions();
```

---

## 📄 文档加载 (Document Loader)

**多格式支持，开箱即用！**

```csharp
using SharpAIKit.DocumentLoader;

// 文本文件
var textLoader = new TextFileLoader("document.txt");

// CSV 文件（列感知）
var csvLoader = new CsvLoader("data.csv")
{
    OneDocumentPerRow = true,
    ContentColumns = new[] { "title", "content" },
    MetadataColumns = new[] { "id", "category" }
};

// Markdown（支持按标题分割）
var mdLoader = new MarkdownLoader("readme.md")
{
    SplitByHeaders = true,
    SplitHeaderLevel = 2
};

// 目录批量加载
var dirLoader = new TextDirectoryLoader("./docs", "*.txt", recursive: true);

// 网页加载
var webLoader = new WebLoader("https://example.com/api/data");

// 加载文档
var documents = await csvLoader.LoadAsync();
foreach (var doc in documents)
{
    Console.WriteLine($"内容: {doc.Content}");
    Console.WriteLine($"来源: {doc.Metadata["source"]}");
}
```

---

## 📊 可观测性 (Callback)

**全链路追踪，生产环境必备！**

```csharp
using SharpAIKit.Callback;

// 控制台输出（调试用）
var consoleCallback = new ConsoleCallbackHandler { UseColors = true };

// 日志记录
var loggingCallback = new LoggingCallbackHandler(logger);

// 性能指标
var metricsCallback = new MetricsCallbackHandler();

// 文件持久化
var fileCallback = new FileCallbackHandler("llm_logs.jsonl");

// 回调管理器
var manager = new CallbackManager()
    .AddHandler(consoleCallback)
    .AddHandler(metricsCallback);

// 查看指标
var metrics = metricsCallback.GetSummary();
Console.WriteLine($"调用次数: {metrics.LLMCalls}");
Console.WriteLine($"平均延迟: {metrics.AverageLatencyMs}ms");
Console.WriteLine($"总 Token: {metrics.TotalTokens}");
```

---

## 🖼️ 多模态 (MultiModal)

**图像理解，Vision 支持！**

```csharp
using SharpAIKit.MultiModal;

// 从 URL 创建图像消息
var message = MultiModalMessage.User(
    "这张图片里有什么？",
    "https://example.com/image.jpg"
);

// Fluent 构建器
var multiModal = new MultiModalMessageBuilder()
    .WithRole("user")
    .AddText("对比这两张图片:")
    .AddImage("https://example.com/img1.jpg")
    .AddImage("https://example.com/img2.jpg")
    .Build();

// 从本地文件（自动 Base64 编码）
var localImage = MultiModalExtensions.CreateVisionMessageFromFile(
    "详细描述这张图片",
    "./my-image.png"
);
```

---

## 🤖 高级 Agent

**ReAct、Plan-Execute、Multi-Agent 三种模式！**

```csharp
using SharpAIKit.Agent;

// 1. ReAct Agent - 推理 + 行动循环
var reactAgent = new ReActAgent(client)
    .AddTool(new CalculatorTool())
    .AddTool(new WebSearchTool());

var result = await reactAgent.RunAsync("搜索最新的AI新闻并总结");
// 输出包含: Thought -> Action -> Observation 循环

// 2. Plan-and-Execute Agent - 先规划后执行
var planAgent = new PlanAndExecuteAgent(client)
    .AddTool(new CalculatorTool());

var planResult = await planAgent.RunAsync("分析这份数据并生成报告");
// 输出包含: 计划步骤 + 每步执行结果

// 3. Multi-Agent System - 多Agent协作
var multiAgent = new MultiAgentSystem(client)
    .AddAgent("researcher", "研究专家", "你是一个专业的研究人员...")
    .AddAgent("writer", "内容写手", "你是一个优秀的技术写手...")
    .AddAgent("reviewer", "质量审核", "你是一个严谨的审核员...");

var teamResult = await multiAgent.RunAsync("撰写一篇关于AI的技术博客");
// 输出包含: 任务分配 + 各Agent响应 + 综合答案
```

---

## 📚 RAG 引擎

```csharp
using SharpAIKit.RAG;

var rag = new RagEngine(client);

// 索引文档
await rag.IndexDocumentAsync("docs/guide.txt");
await rag.IndexContentAsync("你的文档内容...");
await rag.IndexDirectoryAsync("./knowledge", "*.md");

// 智能问答
var answer = await rag.AskAsync("如何使用这个功能？");

// 流式回答
await foreach (var chunk in rag.AskStreamAsync("什么是 RAG？"))
{
    Console.Write(chunk);
}

// 仅检索（不生成）
var docs = await rag.RetrieveAsync("相关查询", topK: 5);
```

---

## 🔮 Native C# Code Interpreter

**🎯 杀手级功能：让 Agent 直接编写并执行 C# 代码，无需 Python！**

### 为什么这是杀手级功能？

- **痛点**：LangChain 的 Code Interpreter 依赖 Python，部署麻烦、速度慢
- **优势**：利用 .NET 的 Roslyn 编译器，在内存中沙箱运行，速度极快
- **效果**：Agent 不再是"语言巨人，数学矮子"，能写代码算数学、处理数据

### 基础使用

```csharp
using SharpAIKit.CodeInterpreter;

var interpreter = new RoslynCodeInterpreter();

// 数学计算
var mathCode = """
    var a = 3;
    var b = 5;
    var result = Math.Pow(a, b);
    result
    """;
var result = await interpreter.ExecuteAsync<double>(mathCode);
Console.WriteLine($"3^5 = {result}");  // 输出: 3^5 = 243

// 斐波那契数列
var fibCode = """
    var n = 10;
    var fib = new List<int> { 0, 1 };
    for (int i = 2; i < n; i++)
    {
        fib.Add(fib[i-1] + fib[i-2]);
    }
    string.Join(", ", fib)
    """;
var fibResult = await interpreter.ExecuteAsync(fibCode);
Console.WriteLine(fibResult.Output);  // 输出: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
```

### 数据处理

```csharp
// 列表处理
var dataCode = """
    var numbers = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var evens = numbers.Where(n => n % 2 == 0).ToList();
    var sum = evens.Sum();
    var avg = evens.Average();
    $"偶数: {string.Join(", ", evens)}, 和: {sum}, 平均值: {avg}"
    """;
var dataResult = await interpreter.ExecuteAsync(dataCode);
Console.WriteLine(dataResult.Output);
// 输出: 偶数: 2, 4, 6, 8, 10, 和: 30, 平均值: 6
```

### 与 Agent 集成

```csharp
using SharpAIKit.Agent;

// 创建使用 Code Interpreter 的工具
public class CodeInterpreterTool : ToolBase
{
    private readonly ICodeInterpreter _interpreter;

    public CodeInterpreterTool(ICodeInterpreter interpreter)
    {
        _interpreter = interpreter;
    }

    [Tool("execute_code", "执行 C# 代码并返回结果")]
    public async Task<string> ExecuteCode(
        [Parameter("要执行的 C# 代码")] string code)
    {
        var result = await _interpreter.ExecuteAsync(code);
        return result.Success ? result.Output : $"错误: {result.Error}";
    }
}

// 在 Agent 中使用
var agent = new AiAgent(client);
agent.AddTool(new CodeInterpreterTool(interpreter));

var answer = await agent.RunAsync("计算 1 到 100 所有偶数的平方和");
// Agent 会自动编写代码并执行！
```

### 高级特性

```csharp
// 变量持久化（在同一会话中）
interpreter.SetVariable("x", 10);
var code1 = "var y = x * 2; y";  // 使用之前设置的变量
var result1 = await interpreter.ExecuteAsync<int>(code1);

// 获取执行结果
var execResult = await interpreter.ExecuteAsync(code);
Console.WriteLine($"成功: {execResult.Success}");
Console.WriteLine($"输出: {execResult.Output}");
Console.WriteLine($"执行时间: {execResult.ExecutionTimeMs}ms");
Console.WriteLine($"变量: {string.Join(", ", execResult.Variables.Keys)}");

// 重置上下文
interpreter.Reset();
```

### 性能优势

| 特性 | LangChain (Python) | SharpAIKit (C#) |
|:-----|:------------------:|:---------------:|
| 启动时间 | 慢（需要 Python 环境） | 快（原生编译） |
| 执行速度 | 中等 | **极快** |
| 内存占用 | 高 | 低 |
| 部署复杂度 | 高（需要 Python） | **低（无需额外依赖）** |

---

## 🕸️ SharpGraph 图编排

**🎯 杀手级功能：基于有限状态机 (FSM) 的图编排引擎，支持循环和复杂分支！**

### 为什么这是杀手级功能？

- **痛点**：LangChain 的 Chain 是线性的（DAG），很难处理循环（如：写代码 → 运行 → 报错 → 修改 → 再运行）
- **优势**：实现基于有限状态机的图编排，可以定义 State（状态）和 Edge（边），让 Agent 在图中"游走"
- **效果**：轻松构建具备"自我纠错"能力的循环 Agent

### 基础使用

```csharp
using SharpAIKit.Graph;

// 创建一个简单的图
var graph = new SharpGraphBuilder("start", maxIterations: 20)
    .Node("start", async state =>
    {
        Console.WriteLine("开始执行任务");
        state.Set("task", "计算斐波那契数列");
        state.NextNode = "process";
        return state;
    })
    .Node("process", async state =>
    {
        Console.WriteLine("处理任务");
        state.Set("result", "完成");
        state.NextNode = "end";
        return state;
    })
    .Node("end", async state =>
    {
        Console.WriteLine("任务完成");
        state.Output = state.Get<string>("result");
        state.ShouldEnd = true;
        return state;
    })
    .Build();

var result = await graph.ExecuteAsync();
Console.WriteLine($"结果: {result.Output}");
```

### 循环和条件分支

```csharp
// 创建一个"写代码 → 运行 → 检查错误 → 修复"的循环图
var graph = new SharpGraphBuilder("start", maxIterations: 20)
    .Node("start", async state =>
    {
        state.Set("attempts", 0);
        state.NextNode = "write_code";
        return state;
    })
    .Node("write_code", async state =>
    {
        var attempts = state.Get<int>("attempts");
        Console.WriteLine($"尝试 #{attempts + 1}: 编写代码");
        state.Set("code", "var x = 10; x * 2");
        state.NextNode = "execute_code";
        return state;
    })
    .Node("execute_code", async state =>
    {
        var code = state.Get<string>("code");
        try
        {
            // 执行代码
            var result = await interpreter.ExecuteAsync(code);
            state.Set("result", result.Output);
            state.Set("error", (string?)null);
            state.NextNode = "check_result";
        }
        catch (Exception ex)
        {
            state.Set("error", ex.Message);
            state.NextNode = "fix_code";
        }
        return state;
    })
    .Node("check_result", async state =>
    {
        var result = state.Get<string>("result");
        if (!string.IsNullOrEmpty(result))
        {
            state.Output = result;
            state.ShouldEnd = true;
        }
        else
        {
            state.NextNode = "fix_code";
        }
        return state;
    })
    .Node("fix_code", async state =>
    {
        var attempts = state.Get<int>("attempts") + 1;
        state.Set("attempts", attempts);
        
        if (attempts >= 3)
        {
            state.ShouldEnd = true;
            state.Output = "超过最大尝试次数";
        }
        else
        {
            state.NextNode = "write_code";  // 循环回去
        }
        return state;
    })
    .Build();

var finalState = await graph.ExecuteAsync();
```

### 条件边（Conditional Edges）

```csharp
var graph = new SharpGraphBuilder("start")
    .Node("start", async state =>
    {
        state.Set("input", 10);
        return state;
    })
    .Node("check", async state =>
    {
        var input = state.Get<int>("input");
        // 根据条件决定下一个节点
        state.NextNode = input > 5 ? "large" : "small";
        return state;
    })
    .Node("large", async state =>
    {
        state.Output = "输入值较大";
        state.ShouldEnd = true;
        return state;
    })
    .Node("small", async state =>
    {
        state.Output = "输入值较小";
        state.ShouldEnd = true;
        return state;
    })
    // 使用条件边
    .Edge("check", "large", condition: state => state.Get<int>("input") > 5)
    .Edge("check", "small", condition: state => state.Get<int>("input") <= 5)
    .Build();
```

### 可视化图结构

```csharp
// 生成 GraphViz 格式的可视化
var graphViz = graph.GetGraphViz();
Console.WriteLine(graphViz);
// 可以复制到 https://dreampuf.github.io/GraphvizOnline/ 查看图形
```

### 与 LangChain 对比

| 特性 | LangChain Chain | SharpGraph |
|:-----|:---------------:|:----------:|
| 结构 | 线性 DAG | **图（支持循环）** |
| 循环支持 | ❌ | ✅ |
| 条件分支 | 有限 | **强大** |
| 状态管理 | 简单 | **完整** |
| 自纠错能力 | ❌ | ✅ |

---

## 🧬 DSPy-style Optimizer

**🎯 杀手级功能：自动优化提示词，让框架自己写 Prompt，越用越聪明！**

### 为什么这是杀手级功能？

- **痛点**：LangChain 的 Prompt 是写死的字符串，效果不好只能人工改，像"炼丹"
- **优势**：实现类似 DSPy 的机制，定义任务和评估标准，框架自动通过多次迭代优化 Prompt
- **效果**：从 "Help me" 自动优化成 "You are an expert... [Few-Shot Examples]..."

### 基础使用

```csharp
using SharpAIKit.Optimizer;

var optimizer = new DSPyOptimizer(client)
{
    MaxIterations = 10,
    TargetScore = 0.9,
    FewShotExamples = 3
};

// 添加训练示例
optimizer
    .AddExample("什么是 C#?", "C# 是一种由微软开发的面向对象编程语言")
    .AddExample("什么是 Python?", "Python 是一种解释型、面向对象的高级编程语言")
    .AddExample("什么是 Java?", "Java 是一种跨平台的面向对象编程语言");

// 设置评估指标
optimizer.SetMetric(Metrics.Contains);

// 优化提示词
var initialPrompt = "回答关于编程语言的问题: {input}";
var result = await optimizer.OptimizeAsync(initialPrompt);

Console.WriteLine($"优化后的提示词:\n{result.OptimizedPrompt}");
Console.WriteLine($"最佳分数: {result.BestScore:F2}");
Console.WriteLine($"迭代次数: {result.Iterations}");
```

### 评估指标

```csharp
// 1. 精确匹配
optimizer.SetMetric(Metrics.ExactMatch);

// 2. 包含匹配
optimizer.SetMetric(Metrics.Contains);

// 3. 语义相似度（使用嵌入向量）
optimizer.SetMetric(Metrics.SemanticSimilarity(client));

// 4. 自定义指标
optimizer.SetMetric(Metrics.Custom(async (input, output, expected) =>
{
    // 自定义评分逻辑
    var score = 0.0;
    if (output.Contains(expected)) score += 0.5;
    if (output.Length > 50) score += 0.3;
    if (output.Contains("编程语言")) score += 0.2;
    return score;
}));
```

### 优化历史追踪

```csharp
var result = await optimizer.OptimizeAsync(initialPrompt);

// 查看优化历史
foreach (var step in result.History)
{
    Console.WriteLine($"迭代 {step.Iteration}:");
    Console.WriteLine($"  分数: {step.Score:F2}");
    Console.WriteLine($"  提示词: {step.Prompt.Substring(0, Math.Min(50, step.Prompt.Length))}...");
    Console.WriteLine($"  各示例分数: {string.Join(", ", step.ExampleScores.Select(s => s.ToString("F2")))}");
}
```

### 实际应用示例

```csharp
// 优化情感分析提示词
var sentimentOptimizer = new DSPyOptimizer(client)
{
    MaxIterations = 5,
    TargetScore = 0.95
};

sentimentOptimizer
    .AddExample("我喜欢这个产品", "积极")
    .AddExample("太糟糕了", "消极")
    .AddExample("还行，没什么特别的", "中性")
    .AddExample("非常满意！", "积极")
    .AddExample("不推荐购买", "消极");

sentimentOptimizer.SetMetric(Metrics.ExactMatch);

var initialSentimentPrompt = "分析以下文本的情感: {input}";
var optimized = await sentimentOptimizer.OptimizeAsync(initialSentimentPrompt);

// 使用优化后的提示词
var testPrompt = optimized.OptimizedPrompt.Replace("{input}", "这个服务很棒！");
var response = await client.ChatAsync(testPrompt);
```

### 优化策略

优化器会自动：
1. **分析最佳示例**：找出表现最好的示例，学习其模式
2. **识别问题示例**：找出表现最差的示例，分析问题
3. **生成改进版本**：结合最佳实践，生成包含 Few-shot 的优化提示词
4. **迭代优化**：重复上述过程，直到达到目标分数

### 与手动优化对比

| 方式 | 时间 | 效果 | 可重复性 |
|:-----|:----:|:----:|:--------:|
| 手动优化 | 数小时 | 不确定 | ❌ |
| DSPy Optimizer | **几分钟** | **稳定提升** | ✅ |

---

## 🌐 支持的提供商

| 提供商 | Base URL | 预设方法 |
|:-------|:---------|:---------|
| OpenAI | `https://api.openai.com/v1` | `CreateOpenAI()` |
| DeepSeek | `https://api.deepseek.com/v1` | `CreateDeepSeek()` |
| Qwen (通义千问) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `CreateQwen()` |
| Mistral | `https://api.mistral.ai/v1` | `CreateMistral()` |
| Yi (零一万物) | `https://api.lingyiwanwu.com/v1` | `CreateYi()` |
| Groq | `https://api.groq.com/openai/v1` | `CreateGroq()` |
| Moonshot (Kimi) | `https://api.moonshot.cn/v1` | `CreateMoonshot()` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `CreateZhipu()` |
| Ollama (本地) | `http://localhost:11434` | `CreateOllama()` |
| **任何 OpenAI 兼容** | 自定义 | `Create(key, url, model)` |

---

## 📁 项目结构

```
SharpAIKit/
├── 📂 src/SharpAIKit/
│   ├── 📂 LLM/              # LLM 客户端
│   ├── 📂 Chain/            # 链式调用 ⭐ NEW
│   ├── 📂 Memory/           # 对话记忆 ⭐ NEW
│   ├── 📂 Prompt/           # 提示模板 ⭐ NEW
│   ├── 📂 Output/           # 输出解析 ⭐ NEW
│   ├── 📂 DocumentLoader/   # 文档加载 ⭐ NEW
│   ├── 📂 Callback/         # 可观测性 ⭐ NEW
│   ├── 📂 MultiModal/       # 多模态   ⭐ NEW
│   ├── 📂 Agent/            # 智能代理 (含 ReAct/MultiAgent)
│   ├── 📂 RAG/              # RAG 引擎
│   ├── 📂 CodeInterpreter/  # 代码解释器 🔮 杀手级功能
│   ├── 📂 Graph/            # 图编排 🕸️ 杀手级功能
│   └── 📂 Optimizer/        # 自动优化 🧬 杀手级功能
├── 📂 samples/              # 示例项目
└── 📂 tests/                # 单元测试
```

---

## 📄 许可证

本项目基于 **MIT 许可证** 开源。

---

<div align="center">

### 🚀 SharpAIKit - .NET 世界的 LangChain，但更好！

**🎯 三大杀手级功能，LangChain 都没有：**
- 🔮 **Native C# Code Interpreter** - 无需 Python，原生执行 C# 代码
- 🕸️ **SharpGraph** - 支持循环和复杂分支的图编排
- 🧬 **DSPy Optimizer** - 自动优化提示词，越用越聪明

**为 .NET 社区用 ❤️ 打造**

如果这个项目对你有帮助，请点个 ⭐ **Star** 支持！

</div>
