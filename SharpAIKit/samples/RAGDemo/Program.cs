using SharpAIKit.LLM;
using SharpAIKit.RAG;

Console.WriteLine("=== SharpAIKit RAG Demo (DeepSeek) ===\n");

// Use DeepSeek API
var client = LLMClientFactory.CreateDeepSeek("sk-e164311ef7914e46a5d760c505714b94");

// Create RAG Engine
var rag = new RagEngine(client);

try
{
    // Example 1: Index Content Directly
    Console.WriteLine("[Example 1: Index Content]");
    Console.WriteLine("Indexing content...");

    var sampleContent = """
        Bone and Joint Infection Treatment Guidelines

        Osteomyelitis is a bone infection caused by bacteria, fungi, or other germs.
        Treatment usually includes:
        1. Antibiotic therapy: usually requires 4-6 weeks of intravenous antibiotics
        2. Surgical debridement: removal of infected bone tissue
        3. Wound care: regular cleaning and dressing changes

        Septic arthritis is an infection of the joint cavity.
        Treatment includes:
        1. Joint aspiration: draining infected fluid
        2. Antibiotic therapy: targeted treatment based on culture results
        3. Physical therapy: maintaining joint mobility

        Preventive measures:
        - Proper wound care
        - Prompt treatment of skin infections
        - Prophylactic antibiotics before surgery
        """;

    await rag.IndexContentAsync(sampleContent, new Dictionary<string, string>
    {
        ["source"] = "medical_guide",
        ["topic"] = "bone_infections"
    });

    Console.WriteLine($"Content indexed. Total chunks: {await rag.GetDocumentCountAsync()}\n");

    // Example 2: Q&A
    Console.WriteLine("[Example 2: RAG Q&A]");

    var questions = new[]
    {
        "How to treat bone infections?",
        "What is septic arthritis?",
        "What are the preventive measures?"
    };

    foreach (var question in questions)
    {
        Console.WriteLine($"\nQ: {question}");
        Console.Write("A: ");

        await foreach (var chunk in rag.AskStreamAsync(question))
        {
            Console.Write(chunk);
        }
        Console.WriteLine("\n");
    }

    // Example 3: Retrieve Relevant Documents
    Console.WriteLine("[Example 3: Document Retrieval]");
    var searchResults = await rag.RetrieveAsync("Antibiotic therapy", topK: 2);

    Console.WriteLine("Most relevant documents:");
    foreach (var result in searchResults)
    {
        Console.WriteLine($"  Similarity: {result.Score:F4}");
        Console.WriteLine($"  Content: {result.Document.Content[..Math.Min(100, result.Document.Content.Length)]}...\n");
    }

    // Example 4: Index from File
    Console.WriteLine("[Example 4: Index from File]");
    var sampleFilePath = "sample_doc.txt";

    // Create sample file
    await File.WriteAllTextAsync(sampleFilePath, """
        C# Programming Language Overview

        C# is a modern object-oriented programming language developed by Microsoft.
        It is part of the .NET ecosystem and is widely used for building:
        - Desktop applications (Windows Forms, WPF)
        - Web applications (ASP.NET Core)
        - Mobile applications (Xamarin, MAUI)
        - Cloud services (Azure Functions)
        - Game development (Unity)

        Key features of C#:
        - Strong typing
        - Garbage collection
        - LINQ data query
        - async/await asynchronous programming
        - Pattern matching
        """);

    await rag.IndexDocumentAsync(sampleFilePath);
    Console.WriteLine($"File indexed. Total chunks: {await rag.GetDocumentCountAsync()}\n");

    // Ask about new content
    Console.WriteLine("Q: What can be built with C#?");
    Console.Write("A: ");
    var answer = await rag.AskAsync("What can be built with C#?");
    Console.WriteLine(answer);

    // Cleanup
    File.Delete(sampleFilePath);
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
finally
{
    client.Dispose();
}

Console.WriteLine("\n=== RAG Demo Completed ===");
