#!/usr/bin/env python3
"""
Test runner for conversational routing system tests.

This script runs comprehensive tests for the conversational routing system,
including unit tests, integration tests, API tests, and performance tests.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False, markers=None):
    """Run the specified test suite."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=rag_engine.core.conversational_routing", 
                   "--cov=rag_engine.core.conversational_integration",
                   "--cov-report=html", 
                   "--cov-report=term-missing"])
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add markers if specified
    if markers:
        cmd.extend(["-m", markers])
    
    # Determine which tests to run
    test_files = []
    
    if test_type == "all" or test_type == "unit":
        test_files.extend([
            "tests/unit/test_conversational_routing.py",
            "tests/unit/test_conversational_routing_advanced.py"
        ])
    
    if test_type == "all" or test_type == "integration":
        test_files.extend([
            "tests/integration/test_conversational_routing_api.py",
            "tests/integration/test_conversational_routing_e2e.py"
        ])
    
    if test_type == "api":
        test_files.append("tests/integration/test_conversational_routing_api.py")
    
    if test_type == "e2e":
        test_files.append("tests/integration/test_conversational_routing_e2e.py")
    
    if test_type == "performance":
        cmd.extend(["-m", "performance"])
        test_files.extend([
            "tests/unit/test_conversational_routing_advanced.py"
        ])
    
    if test_type == "stress":
        cmd.extend(["-m", "stress"])
        test_files.extend([
            "tests/unit/test_conversational_routing_advanced.py"
        ])
    
    # Add test files to command
    cmd.extend(test_files)
    
    # Filter to only existing files
    existing_files = [f for f in test_files if Path(f).exists()]
    if not existing_files:
        print(f"No test files found for type '{test_type}'")
        return False
    
    # Run the tests
    print(f"Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def run_specific_test_class(test_class, verbose=False):
    """Run a specific test class."""
    cmd = ["python", "-m", "pytest", "-v" if verbose else "-q"]
    
    # Map test class names to files
    test_class_map = {
        "TestConversationalRouter": "tests/unit/test_conversational_routing.py::TestConversationalRouter",
        "TestConversationalRAGPrompter": "tests/unit/test_conversational_routing.py::TestConversationalRAGPrompter",
        "TestConversationalRoutingAPI": "tests/integration/test_conversational_routing_api.py::TestConversationalRoutingAPI",
        "TestConversationalRoutingE2E": "tests/integration/test_conversational_routing_e2e.py::TestConversationalRoutingE2E",
        "TestConversationalRoutingEdgeCases": "tests/unit/test_conversational_routing_advanced.py::TestConversationalRoutingEdgeCases",
        "TestConversationalRoutingPerformance": "tests/unit/test_conversational_routing_advanced.py::TestConversationalRoutingPerformance",
        "TestConversationalRoutingStressTest": "tests/unit/test_conversational_routing_advanced.py::TestConversationalRoutingStressTest"
    }
    
    if test_class not in test_class_map:
        print(f"Unknown test class: {test_class}")
        print(f"Available test classes: {', '.join(test_class_map.keys())}")
        return False
    
    cmd.append(test_class_map[test_class])
    
    print(f"Running test class: {test_class}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running test class: {e}")
        return False


def run_test_suite_report():
    """Run all tests and generate a comprehensive report."""
    print("=" * 80)
    print("CONVERSATIONAL ROUTING TEST SUITE REPORT")
    print("=" * 80)
    
    test_results = {}
    
    # Run different test categories
    test_categories = [
        ("unit", "Unit Tests"),
        ("integration", "Integration Tests"),
        ("api", "API Tests"),
        ("e2e", "End-to-End Tests"),
        ("performance", "Performance Tests")
    ]
    
    for test_type, description in test_categories:
        print(f"\n{description}:")
        print("-" * 40)
        
        success = run_tests(test_type=test_type, verbose=False)
        test_results[test_type] = success
        
        status = "PASSED" if success else "FAILED"
        print(f"{description}: {status}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    failed_tests = total_tests - passed_tests
    
    for test_type, description in test_categories:
        status = "PASSED" if test_results[test_type] else "FAILED"
        print(f"{description:<20}: {status}")
    
    print(f"\nTotal: {total_tests}, Passed: {passed_tests}, Failed: {failed_tests}")
    
    if failed_tests == 0:
        print("🎉 All test suites passed!")
        return True
    else:
        print(f"❌ {failed_tests} test suite(s) failed.")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run conversational routing tests")
    
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "api", "e2e", "performance", "stress", "report"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--class",
        dest="test_class",
        help="Run a specific test class"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    
    parser.add_argument(
        "-m", "--markers",
        help="Run tests with specific markers"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (excludes slow and stress tests)"
    )
    
    args = parser.parse_args()
    
    # Set working directory to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Handle quick mode
    if args.quick:
        args.markers = "not slow and not stress"
    
    # Handle specific test class
    if args.test_class:
        success = run_specific_test_class(args.test_class, args.verbose)
    elif args.test_type == "report":
        success = run_test_suite_report()
    else:
        success = run_tests(
            test_type=args.test_type,
            verbose=args.verbose,
            coverage=args.coverage,
            markers=args.markers
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
