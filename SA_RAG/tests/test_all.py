"""Run all tests"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', '.venv', 'lib', 'python3.10', 'site-packages'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


def main():
    """Run all test suites"""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "SA-RAG Test Suite" + " " * 30 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    results = {}
    
    # Run Rust core tests
    print("Running Rust Core Tests...")
    print("=" * 70)
    try:
        from test_rust_core import run_all_tests as run_rust_tests
        results['rust_core'] = run_rust_tests()
    except Exception as e:
        print(f"✗ Rust core tests failed: {e}")
        results['rust_core'] = False
    print()
    
    # Run Python layer tests
    print("Running Python Layer Tests...")
    print("=" * 70)
    try:
        from test_python_layer import run_all_tests as run_python_tests
        results['python_layer'] = run_python_tests()
    except Exception as e:
        print(f"✗ Python layer tests failed: {e}")
        results['python_layer'] = False
    print()
    
    # Run integration tests
    print("Running Integration Tests...")
    print("=" * 70)
    try:
        from test_integration import run_all_tests as run_integration_tests
        results['integration'] = run_integration_tests()
    except Exception as e:
        print(f"✗ Integration tests failed: {e}")
        results['integration'] = False
    print()
    
    # Summary
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 25 + "Test Summary" + " " * 31 + "║")
    print("╠" + "═" * 68 + "╣")
    
    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"║  {status:6}  {name:20} " + " " * 38 + "║")
    
    print("╠" + "═" * 68 + "╣")
    print(f"║  Total: {total_passed}/{total_tests} test suites passed" + " " * (68 - 40) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    if all(results.values()):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

