# SA-RAG Test Suite

This directory contains comprehensive tests for the SA-RAG framework.

## Running Tests

### Using uv (Recommended)

The easiest way to run tests is using the provided script:

```bash
./run_tests.sh
```

Or manually with uv:

```bash
cd python
uv sync --dev
uv run pytest ../tests/test_framework_validation.py -v
```

### Using pytest directly

If you have pytest installed in your environment:

```bash
pytest tests/test_framework_validation.py -v
```

## Test Files

### `test_framework_validation.py`

Comprehensive framework validation tests that verify:

- ✅ Framework initialization
- ✅ Document indexing
- ✅ Search functionality
- ✅ Q&A functionality
- ✅ Memory management
- ✅ Document updates
- ✅ Next-generation features
- ✅ Error handling
- ✅ Performance characteristics
- ✅ Integration workflows

### `test_framework_simple.py`

Simple test script that doesn't require pytest (for quick validation).

## Test Structure

Tests are organized into classes:

- `TestFrameworkInitialization`: Basic setup and imports
- `TestDocumentIndexing`: Document indexing functionality
- `TestSearchFunctionality`: Search operations
- `TestQAFunctionality`: Question answering
- `TestMemoryManagement`: Memory operations
- `TestDocumentUpdate`: Document updates
- `TestNextGenFeatures`: Next-generation features
- `TestErrorHandling`: Error scenarios
- `TestPerformance`: Performance benchmarks
- `TestIntegration`: End-to-end workflows

## Running Specific Tests

Run a specific test class:

```bash
uv run pytest tests/test_framework_validation.py::TestDocumentIndexing -v
```

Run a specific test:

```bash
uv run pytest tests/test_framework_validation.py::TestDocumentIndexing::test_index_single_document -v
```

## Test Output

Tests will show:
- ✓ Passed tests
- ✗ Failed tests
- ⚠ Skipped tests (when dependencies are missing)

## Requirements

- Python 3.9+
- uv package manager
- pytest (installed via uv)
- SA-RAG Python package (installed in development mode)

## Troubleshooting

### Module not found errors

If you see import errors, make sure the Python package is installed:

```bash
cd python
uv sync
```

### Rust core not available

If `rust_core` is not available, some tests will be skipped. This is expected if the Rust core hasn't been built yet. To build it:

```bash
cd rust_core
maturin develop --release
```
