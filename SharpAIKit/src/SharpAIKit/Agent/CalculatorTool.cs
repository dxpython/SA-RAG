namespace SharpAIKit.Agent;

/// <summary>
/// Calculator tool providing basic mathematical operations.
/// </summary>
public class CalculatorTool : ToolBase
{
    /// <summary>
    /// Evaluates a mathematical expression.
    /// </summary>
    /// <param name="expression">The mathematical expression to evaluate (e.g., '3 + 5 * 2').</param>
    /// <returns>The calculation result.</returns>
    [Tool("calculate", "Evaluates a mathematical expression supporting addition, subtraction, multiplication, division, and power operations")]
    public string Calculate(
        [Parameter("The mathematical expression to calculate, e.g., '3 + 5 * 2'")] string expression)
    {
        try
        {
            var result = EvaluateExpression(expression);
            return $"Result: {expression} = {result}";
        }
        catch (Exception ex)
        {
            return $"Calculation error: {ex.Message}";
        }
    }

    /// <summary>
    /// Calculates a power operation.
    /// </summary>
    /// <param name="baseNumber">The base number.</param>
    /// <param name="exponent">The exponent.</param>
    /// <returns>The result of base raised to the power of exponent.</returns>
    [Tool("power", "Calculates the power of a number (base^exponent)")]
    public string Power(
        [Parameter("The base number")] double baseNumber,
        [Parameter("The exponent")] double exponent)
    {
        var result = Math.Pow(baseNumber, exponent);
        return $"Result: {baseNumber}^{exponent} = {result}";
    }

    /// <summary>
    /// Calculates the square root of a number.
    /// </summary>
    /// <param name="number">The number to find the square root of.</param>
    /// <returns>The square root result.</returns>
    [Tool("sqrt", "Calculates the square root of a number")]
    public string Sqrt(
        [Parameter("The number to find the square root of")] double number)
    {
        if (number < 0)
        {
            return "Error: Cannot calculate the square root of a negative number";
        }
        var result = Math.Sqrt(number);
        return $"Result: √{number} = {result}";
    }

    /// <summary>
    /// Simple expression evaluator using DataTable.Compute.
    /// </summary>
    private static double EvaluateExpression(string expression)
    {
        // Remove spaces
        expression = expression.Replace(" ", "");

        // Use DataTable.Compute for evaluation
        var table = new System.Data.DataTable();
        var result = table.Compute(expression, null);
        return Convert.ToDouble(result);
    }
}

