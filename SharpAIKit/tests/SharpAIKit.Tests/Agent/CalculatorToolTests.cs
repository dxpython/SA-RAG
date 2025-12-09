using SharpAIKit.Agent;
using Xunit;

namespace SharpAIKit.Tests.Agent;

/// <summary>
/// Unit tests for the CalculatorTool class.
/// </summary>
public class CalculatorToolTests
{
    private readonly CalculatorTool _calculator = new();

    [Fact]
    public void Calculate_SimpleAddition_ReturnsCorrectResult()
    {
        // Act
        var result = _calculator.Calculate("2 + 3");

        // Assert
        Assert.Contains("5", result);
    }

    [Fact]
    public void Calculate_ComplexExpression_ReturnsCorrectResult()
    {
        // Act
        var result = _calculator.Calculate("3 + 5 * 2");

        // Assert
        Assert.Contains("13", result);
    }

    [Fact]
    public void Calculate_InvalidExpression_ReturnsError()
    {
        // Act
        var result = _calculator.Calculate("invalid");

        // Assert
        Assert.Contains("error", result.ToLower());
        
    }

    [Fact]
    public void Power_CalculatesCorrectly()
    {
        // Act
        var result = _calculator.Power(3, 5);

        // Assert
        Assert.Contains("243", result);
    }

    [Fact]
    public void Power_ZeroExponent_ReturnsOne()
    {
        // Act
        var result = _calculator.Power(5, 0);

        // Assert
        Assert.Contains("1", result);
    }

    [Fact]
    public void Sqrt_PositiveNumber_ReturnsCorrectResult()
    {
        // Act
        var result = _calculator.Sqrt(144);

        // Assert
        Assert.Contains("12", result);
    }

    [Fact]
    public void Sqrt_NegativeNumber_ReturnsError()
    {
        // Act
        var result = _calculator.Sqrt(-1);

        // Assert
        Assert.Contains("Error", result);
    }

    [Fact]
    public void GetToolDefinitions_ReturnsThreeTools()
    {
        // Act
        var definitions = _calculator.GetToolDefinitions();

        // Assert
        Assert.Equal(3, definitions.Count);
        Assert.Contains(definitions, d => d.Name == "calculate");
        Assert.Contains(definitions, d => d.Name == "power");
        Assert.Contains(definitions, d => d.Name == "sqrt");
    }

    [Fact]
    public async Task ToolDefinition_Execute_WorksCorrectly()
    {
        // Arrange
        var definitions = _calculator.GetToolDefinitions();
        var powerTool = definitions.First(d => d.Name == "power");

        // Act
        var result = await powerTool.Execute!(new Dictionary<string, object?>
        {
            ["baseNumber"] = 2.0,
            ["exponent"] = 10.0
        });

        // Assert
        Assert.Contains("1024", result);
    }
}

