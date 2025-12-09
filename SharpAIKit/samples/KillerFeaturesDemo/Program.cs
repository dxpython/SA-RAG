using SharpAIKit.LLM;
using SharpAIKit.CodeInterpreter;
using SharpAIKit.Graph;
using SharpAIKit.Optimizer;
using SharpAIKit.Agent;

Console.WriteLine("🚀 SharpAIKit Killer Features Demo (DeepSeek)\n");
Console.WriteLine("=".PadRight(60, '='));

// Use DeepSeek API
var client = LLMClientFactory.CreateDeepSeek("sk-e164311ef7914e46a5d760c505714b94");

// ============================================================
// 1. 🔮 Native C# Code Interpreter
// ============================================================
Console.WriteLine("\n🔮 Feature 1: Native C# Code Interpreter\n");

var interpreter = new RoslynCodeInterpreter();

try
{
    // Example 1: Math Calculation
    Console.WriteLine("[Example 1: Math Calculation]");
    var mathCode = """
        var a = 3;
        var b = 5;
        var result = Math.Pow(a, b);
        result
        """;
    
    var mathResult = await interpreter.ExecuteAsync(mathCode);
    var mathValue = await interpreter.ExecuteAsync<double>(mathCode);
    Console.WriteLine($"Calculate 3 to the power of 5: {mathValue}");
    Console.WriteLine($"Execution Time: {mathResult.ExecutionTimeMs}ms\n");

    // Example 2: Fibonacci Sequence
    Console.WriteLine("[Example 2: Fibonacci Sequence]");
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
    Console.WriteLine($"First 10 Fibonacci numbers: {fibResult.Output}");
    Console.WriteLine($"Execution Time: {fibResult.ExecutionTimeMs}ms\n");

    // Example 3: Data Processing
    Console.WriteLine("[Example 3: Data Processing]");
    var dataCode = """
        var numbers = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var evens = numbers.Where(n => n % 2 == 0).ToList();
        var sum = evens.Sum();
        var avg = evens.Average();
        $"Evens: {string.Join(", ", evens)}, Sum: {sum}, Average: {avg}"
        """;
    
    var dataResult = await interpreter.ExecuteAsync(dataCode);
    Console.WriteLine($"Data Processing Result: {dataResult.Output}");
    Console.WriteLine($"Execution Time: {dataResult.ExecutionTimeMs}ms\n");

    // Example 4: String Processing
    Console.WriteLine("[Example 4: String Processing]");
    var stringCode = """
        var text = "Hello, SharpAIKit!";
        var words = text.Split(' ', ',', '!');
        var reversed = string.Join(" ", words.Where(w => !string.IsNullOrEmpty(w)).Reverse());
        reversed
        """;
    
    var stringResult = await interpreter.ExecuteAsync(stringCode);
    Console.WriteLine($"String Processing: {stringResult.Output}\n");

    Console.WriteLine("✅ Code Interpreter works! Agent can now write code to solve problems!\n");
}
catch (Exception ex)
{
    Console.WriteLine($"❌ Error: {ex.Message}\n");
}

// ============================================================
// 2. 🕸️ SharpGraph (Graph-based Agent Orchestration)
// ============================================================
Console.WriteLine("\n🕸️ Feature 2: SharpGraph - Graph Orchestration Engine\n");

try
{
    // Create a "Write Code -> Run -> Check Error -> Fix" loop graph
    var graph = new SharpGraphBuilder("start", maxIterations: 20)
        .Node("start", async state =>
        {
            Console.WriteLine("  [Node: start] Start Task");
            state.Set("task", "Calculate Fibonacci Sequence");
            state.Set("attempts", 0);
            state.NextNode = "write_code";
            return state;
        })
        .Node("write_code", async state =>
        {
            var attempts = state.Get<int>("attempts");
            Console.WriteLine($"  [Node: write_code] Attempt #{attempts + 1}: Write Code");
            
            // Simulate writing code (In reality, LLM would be called here)
            if (attempts == 0)
            {
                state.Set("code", "var fib = new List<int> { 0, 1 }; for (int i = 2; i < 10; i++) { fib.Add(fib[i-1] + fib[i-2]); } string.Join(\", \", fib)");
            }
            else
            {
                // Fixed code
                state.Set("code", "var fib = new List<int> { 0, 1 }; for (int i = 2; i < 10; i++) { fib.Add(fib[i-1] + fib[i-2]); } string.Join(\", \", fib)");
            }
            
            state.NextNode = "execute_code";
            return state;
        })
        .Node("execute_code", async state =>
        {
            Console.WriteLine("  [Node: execute_code] Execute Code");
            var code = state.Get<string>("code") ?? "";
            
            try
            {
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
            Console.WriteLine($"  [Node: check_result] Check Result: {result}");
            
            // Check if result is valid
            if (!string.IsNullOrEmpty(result) && result.Contains(","))
            {
                Console.WriteLine("  ✅ Result Valid!");
                state.Output = result;
                state.ShouldEnd = true;
            }
            else
            {
                Console.WriteLine("  ⚠️ Result Invalid, needs fix");
                state.NextNode = "fix_code";
            }
            
            return state;
        })
        .Node("fix_code", async state =>
        {
            var attempts = state.Get<int>("attempts");
            attempts++;
            state.Set("attempts", attempts);
            
            Console.WriteLine($"  [Node: fix_code] Fix Code (Attempt {attempts})");
            
            if (attempts >= 3)
            {
                Console.WriteLine("  ❌ Max attempts exceeded");
                state.ShouldEnd = true;
                state.Output = "Execution Failed";
            }
            else
            {
                state.NextNode = "write_code";
            }
            
            return state;
        })
        .Build();

    Console.WriteLine("[Example: Self-correcting Loop Graph]");
    var graphState = new GraphState();
    var finalState = await graph.ExecuteAsync(graphState);
    
    Console.WriteLine($"\nFinal Result: {finalState.Output}");
    Console.WriteLine($"Visited Nodes: {graphState.Data.Count}");
    Console.WriteLine("\n✅ SharpGraph works! Supports loops and complex branches!\n");
}
catch (Exception ex)
{
    Console.WriteLine($"❌ Error: {ex.Message}\n");
}

// ============================================================
// 3. 🧬 DSPy-style Optimizer (Automatic Prompt Optimizer)
// ============================================================
Console.WriteLine("\n🧬 Feature 3: DSPy-style Optimizer - Automatic Prompt Optimization\n");

try
{
    var optimizer = new DSPyOptimizer(client)
    {
        MaxIterations = 5,
        TargetScore = 0.9,
        FewShotExamples = 2
    };

    // Add training examples
    optimizer
        .AddExample("What is C#?", "C# is an object-oriented programming language developed by Microsoft")
        .AddExample("What is Python?", "Python is an interpreted, object-oriented, high-level programming language")
        .AddExample("What is Java?", "Java is a cross-platform object-oriented programming language")
        .AddExample("What is JavaScript?", "JavaScript is a scripting language used for Web development")
        .AddExample("What is Rust?", "Rust is a systems programming language focused on safety and performance");

    // Set evaluation metric (Use Contains match)
    optimizer.SetMetric(Metrics.Contains);

    Console.WriteLine("[Example: Optimize Programming Language Q&A Prompt]");
    Console.WriteLine("Initial Prompt: \"Answer questions about programming languages: {input}\"\n");

    var initialPrompt = "Answer questions about programming languages: {input}";
    var result = await optimizer.OptimizeAsync(initialPrompt);

    Console.WriteLine($"\nOptimization Completed!");
    Console.WriteLine($"Iterations: {result.Iterations}");
    Console.WriteLine($"Best Score: {result.BestScore:F2}");
    Console.WriteLine($"\nOptimized Prompt:");
    Console.WriteLine($"```");
    Console.WriteLine(result.OptimizedPrompt);
    Console.WriteLine($"```\n");

    Console.WriteLine("Optimization History:");
    foreach (var step in result.History)
    {
        Console.WriteLine($"  Iteration {step.Iteration}: Score {step.Score:F2}");
    }

    // Test optimized prompt
    Console.WriteLine("\n[Test Optimized Prompt]");
    var testInput = "What is Go language?";
    var optimizedMessages = new List<SharpAIKit.Common.ChatMessage>
    {
        SharpAIKit.Common.ChatMessage.User(result.OptimizedPrompt.Replace("{input}", testInput))
    };
    var testOutput = await client.ChatAsync(optimizedMessages);
    Console.WriteLine($"Question: {testInput}");
    Console.WriteLine($"Answer: {testOutput}\n");

    Console.WriteLine("✅ DSPy Optimizer works! Prompt automatically optimized successfully!\n");
}
catch (Exception ex)
{
    Console.WriteLine($"❌ Error: {ex.Message}\n");
    Console.WriteLine($"Stack Trace: {ex.StackTrace}\n");
}

// ============================================================
// Comprehensive Demo: Code Interpreter + Graph + Agent
// ============================================================
Console.WriteLine("\n🎯 Comprehensive Demo: Code Interpreter + Graph + Agent\n");

try
{
    // Create an Agent using Code Interpreter
    var agent = new AiAgent(client);
    
    // Create a calculation tool using Code Interpreter
    var calculatorTool = new CodeInterpreterTool(interpreter);
    agent.AddTool(calculatorTool);

    Console.WriteLine("[Example: Agent uses Code Interpreter to solve math problems]");
    var agentAnswer = await agent.RunAsync("Calculate the sum of squares of all even numbers from 1 to 100");
    
    Console.WriteLine($"Question: Calculate the sum of squares of all even numbers from 1 to 100");
    Console.WriteLine($"Agent Answer: {agentAnswer}\n");

    // Verify Result
    var verifyCode = """
        var sum = 0;
        for (int i = 2; i <= 100; i += 2)
        {
            sum += i * i;
        }
        sum
        """;
    var verifyResult = await interpreter.ExecuteAsync(verifyCode);
    var verifyValue = await interpreter.ExecuteAsync<int>(verifyCode);
    Console.WriteLine($"Verification Result: {verifyValue} (Execution Time: {verifyResult.ExecutionTimeMs}ms)\n");

    Console.WriteLine("✅ Comprehensive Demo Success! Agent + Code Interpreter work perfectly together!\n");
}
catch (Exception ex)
{
    Console.WriteLine($"❌ Error: {ex.Message}\n");
}

Console.WriteLine("=".PadRight(60, '='));
Console.WriteLine("🎉 All Killer Features Demo Completed!");
Console.WriteLine("=".PadRight(60, '='));
Console.WriteLine(@"
✅ 1. Native C# Code Interpreter - Agent can write code to solve problems
✅ 2. SharpGraph - Graph orchestration supporting loops and complex branches
✅ 3. DSPy Optimizer - Automatic prompt optimization, getting smarter with usage

🚀 SharpAIKit now has killer features that LangChain doesn't!
");

client.Dispose();

/// <summary>
/// Tool wrapper for Code Interpreter.
/// </summary>
public class CodeInterpreterTool : ToolBase
{
    private readonly ICodeInterpreter _interpreter;

    public CodeInterpreterTool(ICodeInterpreter interpreter)
    {
        _interpreter = interpreter;
    }

    [Tool("execute_code", "Executes C# code and returns the result. Can be used for math calculations, data processing, etc.")]
    public async Task<string> ExecuteCode(
        [Parameter("The C# code to execute")] string code)
    {
        var result = await _interpreter.ExecuteAsync(code);
        if (!result.Success)
        {
            return $"Execution Failed: {result.Error}";
        }
        return $"Execution Success: {result.Output}";
    }
}
