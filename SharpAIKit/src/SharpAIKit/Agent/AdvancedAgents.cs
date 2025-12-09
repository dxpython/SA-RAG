using System.Text;
using System.Text.Json;
using SharpAIKit.Common;
using SharpAIKit.LLM;

namespace SharpAIKit.Agent;

/// <summary>
/// ReAct (Reasoning + Acting) Agent implementation.
/// Uses a thought-action-observation loop for complex reasoning.
/// </summary>
public class ReActAgent
{
    private readonly ILLMClient _llmClient;
    private readonly Dictionary<string, ToolDefinition> _tools = new();

    /// <summary>
    /// Gets or sets the maximum iterations.
    /// </summary>
    public int MaxIterations { get; set; } = 10;

    /// <summary>
    /// Gets or sets the system prompt.
    /// </summary>
    public string SystemPrompt { get; set; } = """
        You are an AI assistant that uses the ReAct framework to solve problems.
        
        Available tools:
        {tools}
        
        For each step, respond in this exact format:
        
        Thought: [Your reasoning about what to do next]
        Action: [tool_name]
        Action Input: [JSON input for the tool]
        
        After receiving an observation, continue with another Thought/Action/Action Input.
        
        When you have enough information to answer, respond with:
        Thought: [Your final reasoning]
        Final Answer: [Your complete answer to the user]
        
        Always start with a Thought. Never skip the Thought step.
        """;

    /// <summary>
    /// Creates a new ReAct agent.
    /// </summary>
    public ReActAgent(ILLMClient llmClient)
    {
        _llmClient = llmClient ?? throw new ArgumentNullException(nameof(llmClient));
    }

    /// <summary>
    /// Adds a tool to the agent.
    /// </summary>
    public ReActAgent AddTool(ToolBase tool)
    {
        foreach (var def in tool.GetToolDefinitions())
        {
            _tools[def.Name] = def;
        }
        return this;
    }

    /// <summary>
    /// Runs the agent on a task.
    /// </summary>
    public async Task<ReActResult> RunAsync(string task, CancellationToken cancellationToken = default)
    {
        var result = new ReActResult();
        var scratchpad = new StringBuilder();
        
        var systemPrompt = BuildSystemPrompt();
        
        for (var i = 0; i < MaxIterations; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var messages = new List<ChatMessage>
            {
                ChatMessage.System(systemPrompt),
                ChatMessage.User($"Task: {task}\n\n{scratchpad}")
            };

            var response = await _llmClient.ChatAsync(messages, cancellationToken);
            
            var step = ParseReActStep(response);
            result.Steps.Add(step);
            
            if (step.IsFinalAnswer)
            {
                result.Answer = step.FinalAnswer ?? string.Empty;
                result.Success = true;
                return result;
            }

            // Execute the action
            if (!string.IsNullOrEmpty(step.Action))
            {
                var observation = await ExecuteToolAsync(step.Action, step.ActionInput);
                step.Observation = observation;
                
                scratchpad.AppendLine($"Thought: {step.Thought}");
                scratchpad.AppendLine($"Action: {step.Action}");
                scratchpad.AppendLine($"Action Input: {step.ActionInput}");
                scratchpad.AppendLine($"Observation: {observation}");
                scratchpad.AppendLine();
            }
        }

        result.Answer = "Maximum iterations reached without finding an answer.";
        result.Success = false;
        return result;
    }

    private string BuildSystemPrompt()
    {
        var toolsDesc = new StringBuilder();
        foreach (var tool in _tools.Values)
        {
            toolsDesc.AppendLine($"- {tool.Name}: {tool.Description}");
            foreach (var param in tool.Parameters)
            {
                toolsDesc.AppendLine($"  - {param.Name} ({param.Type}): {param.Description}");
            }
        }
        return SystemPrompt.Replace("{tools}", toolsDesc.ToString());
    }

    private static ReActStep ParseReActStep(string response)
    {
        var step = new ReActStep();
        var lines = response.Split('\n');

        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            
            if (trimmed.StartsWith("Thought:", StringComparison.OrdinalIgnoreCase))
            {
                step.Thought = trimmed.Substring(8).Trim();
            }
            else if (trimmed.StartsWith("Action:", StringComparison.OrdinalIgnoreCase))
            {
                step.Action = trimmed.Substring(7).Trim();
            }
            else if (trimmed.StartsWith("Action Input:", StringComparison.OrdinalIgnoreCase))
            {
                step.ActionInput = trimmed.Substring(13).Trim();
            }
            else if (trimmed.StartsWith("Final Answer:", StringComparison.OrdinalIgnoreCase))
            {
                step.FinalAnswer = trimmed.Substring(13).Trim();
                step.IsFinalAnswer = true;
            }
        }

        return step;
    }

    private async Task<string> ExecuteToolAsync(string toolName, string? input)
    {
        if (!_tools.TryGetValue(toolName, out var tool))
        {
            return $"Error: Tool '{toolName}' not found.";
        }

        try
        {
            var args = new Dictionary<string, object?>();
            if (!string.IsNullOrEmpty(input))
            {
                try
                {
                    args = JsonSerializer.Deserialize<Dictionary<string, object?>>(input) ?? new();
                }
                catch
                {
                    // If not JSON, try to use as single parameter
                    if (tool.Parameters.Count > 0)
                    {
                        args[tool.Parameters[0].Name] = input;
                    }
                }
            }

            return tool.Execute != null 
                ? await tool.Execute(args) 
                : "Tool has no execute method.";
        }
        catch (Exception ex)
        {
            return $"Error executing tool: {ex.Message}";
        }
    }
}

/// <summary>
/// Result of a ReAct agent execution.
/// </summary>
public class ReActResult
{
    /// <summary>Gets or sets the final answer.</summary>
    public string Answer { get; set; } = string.Empty;
    /// <summary>Gets or sets whether execution was successful.</summary>
    public bool Success { get; set; }
    /// <summary>Gets or sets the execution steps.</summary>
    public List<ReActStep> Steps { get; set; } = new();
}

/// <summary>
/// A single step in ReAct execution.
/// </summary>
public class ReActStep
{
    /// <summary>Gets or sets the thought.</summary>
    public string? Thought { get; set; }
    /// <summary>Gets or sets the action (tool name).</summary>
    public string? Action { get; set; }
    /// <summary>Gets or sets the action input.</summary>
    public string? ActionInput { get; set; }
    /// <summary>Gets or sets the observation (tool result).</summary>
    public string? Observation { get; set; }
    /// <summary>Gets or sets the final answer.</summary>
    public string? FinalAnswer { get; set; }
    /// <summary>Gets or sets whether this is the final step.</summary>
    public bool IsFinalAnswer { get; set; }
}

/// <summary>
/// Plan-and-Execute Agent that creates a plan before executing.
/// </summary>
public class PlanAndExecuteAgent
{
    private readonly ILLMClient _llmClient;
    private readonly Dictionary<string, ToolDefinition> _tools = new();

    /// <summary>
    /// Gets or sets the planning prompt.
    /// </summary>
    public string PlanningPrompt { get; set; } = """
        Given the following task, create a step-by-step plan to accomplish it.
        Each step should be a clear, actionable item.
        
        Available tools:
        {tools}
        
        Task: {task}
        
        Respond with a numbered list of steps:
        1. [First step]
        2. [Second step]
        ...
        
        Only include steps that are necessary. Be concise.
        """;

    /// <summary>
    /// Gets or sets the execution prompt.
    /// </summary>
    public string ExecutionPrompt { get; set; } = """
        You are executing step {step_number} of a plan.
        
        Overall task: {task}
        
        Full plan:
        {plan}
        
        Previous results:
        {previous_results}
        
        Current step to execute: {current_step}
        
        Available tools:
        {tools}
        
        Execute this step. If you need to use a tool, respond with:
        {"tool": "tool_name", "input": {"param": "value"}}
        
        If no tool is needed, just provide the result directly.
        """;

    /// <summary>
    /// Creates a new Plan-and-Execute agent.
    /// </summary>
    public PlanAndExecuteAgent(ILLMClient llmClient)
    {
        _llmClient = llmClient ?? throw new ArgumentNullException(nameof(llmClient));
    }

    /// <summary>
    /// Adds a tool to the agent.
    /// </summary>
    public PlanAndExecuteAgent AddTool(ToolBase tool)
    {
        foreach (var def in tool.GetToolDefinitions())
        {
            _tools[def.Name] = def;
        }
        return this;
    }

    /// <summary>
    /// Runs the agent on a task.
    /// </summary>
    public async Task<PlanExecuteResult> RunAsync(string task, CancellationToken cancellationToken = default)
    {
        var result = new PlanExecuteResult { Task = task };

        // Step 1: Create the plan
        var plan = await CreatePlanAsync(task, cancellationToken);
        result.Plan = plan;

        // Step 2: Execute each step
        var previousResults = new List<string>();
        
        for (var i = 0; i < plan.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var stepResult = await ExecuteStepAsync(
                task, plan, i, previousResults, cancellationToken);
            
            result.StepResults.Add(stepResult);
            previousResults.Add($"Step {i + 1}: {stepResult}");
        }

        // Step 3: Synthesize final answer
        result.Answer = await SynthesizeAnswerAsync(task, plan, previousResults, cancellationToken);
        result.Success = true;
        
        return result;
    }

    private async Task<List<string>> CreatePlanAsync(string task, CancellationToken cancellationToken)
    {
        var prompt = PlanningPrompt
            .Replace("{tools}", FormatTools())
            .Replace("{task}", task);

        var response = await _llmClient.ChatAsync(prompt, cancellationToken);
        
        // Parse numbered list
        var steps = new List<string>();
        var lines = response.Split('\n');
        
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (trimmed.Length > 2 && char.IsDigit(trimmed[0]) && (trimmed[1] == '.' || trimmed[1] == ')'))
            {
                steps.Add(trimmed.Substring(2).Trim());
            }
        }
        
        return steps;
    }

    private async Task<string> ExecuteStepAsync(
        string task, 
        List<string> plan, 
        int stepIndex, 
        List<string> previousResults,
        CancellationToken cancellationToken)
    {
        var prompt = ExecutionPrompt
            .Replace("{step_number}", (stepIndex + 1).ToString())
            .Replace("{task}", task)
            .Replace("{plan}", string.Join("\n", plan.Select((s, i) => $"{i + 1}. {s}")))
            .Replace("{previous_results}", string.Join("\n", previousResults))
            .Replace("{current_step}", plan[stepIndex])
            .Replace("{tools}", FormatTools());

        var response = await _llmClient.ChatAsync(prompt, cancellationToken);

        // Check if it's a tool call
        if (response.Contains("\"tool\""))
        {
            try
            {
                var jsonStart = response.IndexOf('{');
                var jsonEnd = response.LastIndexOf('}');
                if (jsonStart >= 0 && jsonEnd > jsonStart)
                {
                    var json = response.Substring(jsonStart, jsonEnd - jsonStart + 1);
                    using var doc = JsonDocument.Parse(json);
                    
                    var toolName = doc.RootElement.GetProperty("tool").GetString();
                    var input = doc.RootElement.GetProperty("input");
                    
                    if (!string.IsNullOrEmpty(toolName) && _tools.TryGetValue(toolName, out var tool))
                    {
                        var args = new Dictionary<string, object?>();
                        foreach (var prop in input.EnumerateObject())
                        {
                            args[prop.Name] = prop.Value.Clone();
                        }
                        
                        return tool.Execute != null 
                            ? await tool.Execute(args) 
                            : "Tool has no execute method.";
                    }
                }
            }
            catch
            {
                // If parsing fails, return the response as-is
            }
        }

        return response;
    }

    private async Task<string> SynthesizeAnswerAsync(
        string task,
        List<string> plan,
        List<string> results,
        CancellationToken cancellationToken)
    {
        var prompt = $"""
            Given the following task and execution results, provide a final comprehensive answer.
            
            Task: {task}
            
            Execution Results:
            {string.Join("\n", results)}
            
            Final Answer:
            """;

        return await _llmClient.ChatAsync(prompt, cancellationToken);
    }

    private string FormatTools()
    {
        var sb = new StringBuilder();
        foreach (var tool in _tools.Values)
        {
            sb.AppendLine($"- {tool.Name}: {tool.Description}");
        }
        return sb.ToString();
    }
}

/// <summary>
/// Result of Plan-and-Execute agent.
/// </summary>
public class PlanExecuteResult
{
    /// <summary>Gets or sets the original task.</summary>
    public string Task { get; set; } = string.Empty;
    /// <summary>Gets or sets the plan steps.</summary>
    public List<string> Plan { get; set; } = new();
    /// <summary>Gets or sets the step results.</summary>
    public List<string> StepResults { get; set; } = new();
    /// <summary>Gets or sets the final answer.</summary>
    public string Answer { get; set; } = string.Empty;
    /// <summary>Gets or sets whether execution was successful.</summary>
    public bool Success { get; set; }
}

/// <summary>
/// Multi-Agent system for collaborative problem solving.
/// </summary>
public class MultiAgentSystem
{
    private readonly Dictionary<string, AgentRole> _agents = new();
    private readonly ILLMClient _llmClient;

    /// <summary>
    /// Gets or sets the coordinator prompt.
    /// </summary>
    public string CoordinatorPrompt { get; set; } = """
        You are coordinating multiple AI agents to solve a problem.
        
        Available agents:
        {agents}
        
        Task: {task}
        
        Decide which agent should handle this task or delegate subtasks to multiple agents.
        Respond with JSON:
        {"agent": "agent_name", "subtask": "specific task for this agent"}
        
        Or for multiple agents:
        {"delegations": [{"agent": "name1", "subtask": "task1"}, {"agent": "name2", "subtask": "task2"}]}
        """;

    /// <summary>
    /// Creates a new multi-agent system.
    /// </summary>
    public MultiAgentSystem(ILLMClient llmClient)
    {
        _llmClient = llmClient ?? throw new ArgumentNullException(nameof(llmClient));
    }

    /// <summary>
    /// Adds an agent to the system.
    /// </summary>
    public MultiAgentSystem AddAgent(string name, string role, string systemPrompt)
    {
        _agents[name] = new AgentRole
        {
            Name = name,
            Role = role,
            SystemPrompt = systemPrompt
        };
        return this;
    }

    /// <summary>
    /// Runs the multi-agent system on a task.
    /// </summary>
    public async Task<MultiAgentResult> RunAsync(string task, CancellationToken cancellationToken = default)
    {
        var result = new MultiAgentResult { Task = task };

        // Coordinator decides delegation
        var delegation = await GetDelegationAsync(task, cancellationToken);
        result.Delegations = delegation;

        // Execute each delegation
        var responses = new List<(string Agent, string Response)>();
        
        foreach (var (agentName, subtask) in delegation)
        {
            if (_agents.TryGetValue(agentName, out var agent))
            {
                var response = await ExecuteAgentAsync(agent, subtask, cancellationToken);
                responses.Add((agentName, response));
                result.AgentResponses.Add(new AgentResponse
                {
                    AgentName = agentName,
                    Subtask = subtask,
                    Response = response
                });
            }
        }

        // Synthesize final answer
        result.Answer = await SynthesizeAsync(task, responses, cancellationToken);
        result.Success = true;

        return result;
    }

    private async Task<List<(string Agent, string Subtask)>> GetDelegationAsync(
        string task, CancellationToken cancellationToken)
    {
        var agentsDesc = string.Join("\n", _agents.Values.Select(a => $"- {a.Name}: {a.Role}"));
        var prompt = CoordinatorPrompt
            .Replace("{agents}", agentsDesc)
            .Replace("{task}", task);

        var response = await _llmClient.ChatAsync(prompt, cancellationToken);
        var delegations = new List<(string, string)>();

        try
        {
            var jsonStart = response.IndexOf('{');
            var jsonEnd = response.LastIndexOf('}');
            if (jsonStart >= 0 && jsonEnd > jsonStart)
            {
                var json = response.Substring(jsonStart, jsonEnd - jsonStart + 1);
                using var doc = JsonDocument.Parse(json);

                if (doc.RootElement.TryGetProperty("delegations", out var delegationsArray))
                {
                    foreach (var del in delegationsArray.EnumerateArray())
                    {
                        var agent = del.GetProperty("agent").GetString() ?? "";
                        var subtask = del.GetProperty("subtask").GetString() ?? "";
                        delegations.Add((agent, subtask));
                    }
                }
                else if (doc.RootElement.TryGetProperty("agent", out var agentProp))
                {
                    var agent = agentProp.GetString() ?? "";
                    var subtask = doc.RootElement.GetProperty("subtask").GetString() ?? "";
                    delegations.Add((agent, subtask));
                }
            }
        }
        catch
        {
            // Default to first agent with full task
            if (_agents.Count > 0)
            {
                delegations.Add((_agents.Keys.First(), task));
            }
        }

        return delegations;
    }

    private async Task<string> ExecuteAgentAsync(
        AgentRole agent, string subtask, CancellationToken cancellationToken)
    {
        var messages = new List<ChatMessage>
        {
            ChatMessage.System(agent.SystemPrompt),
            ChatMessage.User(subtask)
        };

        return await _llmClient.ChatAsync(messages, cancellationToken);
    }

    private async Task<string> SynthesizeAsync(
        string task, 
        List<(string Agent, string Response)> responses,
        CancellationToken cancellationToken)
    {
        var responsesText = string.Join("\n\n", 
            responses.Select(r => $"[{r.Agent}]:\n{r.Response}"));

        var prompt = $"""
            Synthesize the following agent responses into a comprehensive answer.
            
            Original Task: {task}
            
            Agent Responses:
            {responsesText}
            
            Final Synthesized Answer:
            """;

        return await _llmClient.ChatAsync(prompt, cancellationToken);
    }

    private class AgentRole
    {
        public string Name { get; set; } = string.Empty;
        public string Role { get; set; } = string.Empty;
        public string SystemPrompt { get; set; } = string.Empty;
    }
}

/// <summary>
/// Result of multi-agent execution.
/// </summary>
public class MultiAgentResult
{
    /// <summary>Gets or sets the original task.</summary>
    public string Task { get; set; } = string.Empty;
    /// <summary>Gets or sets the delegations.</summary>
    public List<(string Agent, string Subtask)> Delegations { get; set; } = new();
    /// <summary>Gets or sets the agent responses.</summary>
    public List<AgentResponse> AgentResponses { get; set; } = new();
    /// <summary>Gets or sets the final answer.</summary>
    public string Answer { get; set; } = string.Empty;
    /// <summary>Gets or sets whether execution was successful.</summary>
    public bool Success { get; set; }
}

/// <summary>
/// Response from a single agent.
/// </summary>
public class AgentResponse
{
    /// <summary>Gets or sets the agent name.</summary>
    public string AgentName { get; set; } = string.Empty;
    /// <summary>Gets or sets the subtask.</summary>
    public string Subtask { get; set; } = string.Empty;
    /// <summary>Gets or sets the response.</summary>
    public string Response { get; set; } = string.Empty;
}

