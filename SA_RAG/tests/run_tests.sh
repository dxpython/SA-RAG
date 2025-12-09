#!/bin/bash
# Run SA-RAG framework validation tests using uv

set -e

echo "=========================================="
echo "SA-RAG Framework Validation Tests"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv is installed"
echo ""

# Navigate to python directory
cd "$(dirname "$0")/python" || exit 1

echo "Installing dependencies with uv..."
uv sync --dev

echo ""
echo "Running tests with pytest..."
echo ""

# Run tests
uv run pytest ../tests/test_framework_validation.py -v --tb=short

echo ""
echo "=========================================="
echo "Tests completed!"
echo "=========================================="

