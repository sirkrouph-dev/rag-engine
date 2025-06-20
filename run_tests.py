#!/usr/bin/env python3
"""
Test runner script for RAG Engine tests.
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False):
    """Run the specified type of tests."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.extend(["-v"])
    else:
        cmd.extend(["-q"])
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=rag_engine", "--cov-report=html", "--cov-report=term"])
    
    # Select test type
    if test_type == "unit":
        cmd.extend(["tests/unit"])
    elif test_type == "integration":
        cmd.extend(["tests/integration"])
    elif test_type == "all":
        cmd.extend(["tests/"])
    elif test_type == "fast":
        cmd.extend(["tests/", "-m", "not slow"])
    elif test_type == "no-api":
        cmd.extend(["tests/", "-m", "not requires_api_key and not requires_internet"])
    else:
        print(f"Unknown test type: {test_type}")
        return 1
    
    # Add color output
    cmd.extend(["--color=yes"])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 130


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="RAG Engine Test Runner")
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["unit", "integration", "all", "fast", "no-api"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    if project_root.name == "tests":
        project_root = project_root.parent
    
    print(f"Running tests from: {project_root}")
    
    # Change to project directory
    import os
    os.chdir(project_root)
    
    # Run tests
    return run_tests(args.test_type, args.verbose, args.coverage)


if __name__ == "__main__":
    sys.exit(main())
