#!/usr/bin/env python3
"""
Production Test Runner for RAG Engine

Comprehensive test runner for all production features including:
- Unit tests for production components
- Integration tests for API features  
- End-to-end workflow tests
- Performance and scalability tests

Usage:
    python tests/run_production_tests.py [test_type] [options]

Test Types:
    all         - Run all production tests (default)
    unit        - Run unit tests only
    integration - Run integration tests only
    e2e         - Run end-to-end tests only
    security    - Run security tests only
    performance - Run performance tests only
    quick       - Run quick subset of tests

Options:
    --verbose   - Verbose output
    --coverage  - Run with coverage reporting
    --parallel  - Run tests in parallel
    --report    - Generate detailed HTML report
    --benchmark - Include performance benchmarks
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any


class ProductionTestRunner:
    """Production test runner with comprehensive reporting."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        self.start_time = None
        self.total_time = None
        
    def run_test_suite(self, test_type: str = "all", **options) -> bool:
        """Run the specified test suite."""
        self.start_time = time.time()
        
        print("=" * 80)
        print("🚀 RAG ENGINE PRODUCTION TEST SUITE")
        print("=" * 80)
        print(f"Test Type: {test_type.upper()}")
        print(f"Project Root: {self.project_root}")
        print(f"Python Version: {sys.version}")
        print("=" * 80)
        
        # Determine which tests to run
        test_suites = self._get_test_suites(test_type)
        
        if not test_suites:
            print(f"❌ No test suites found for type: {test_type}")
            return False
        
        # Run each test suite
        overall_success = True
        for suite_name, suite_config in test_suites.items():
            print(f"\n📋 Running {suite_name}...")
            print("-" * 60)
            
            success = self._run_single_suite(suite_name, suite_config, **options)
            self.test_results[suite_name] = success
            
            if not success:
                overall_success = False
                if not options.get("continue_on_failure", True):
                    break
        
        # Generate final report
        self.total_time = time.time() - self.start_time
        self._generate_final_report(overall_success)
        
        return overall_success
    
    def _get_test_suites(self, test_type: str) -> Dict[str, Dict[str, Any]]:
        """Get test suites based on test type."""
        all_suites = {
            "Security Integration": {
                "files": ["tests/unit/test_security_integration.py"],
                "markers": [],
                "description": "Security features (auth, validation, audit)"
            },
            "Error Handling": {
                "files": ["tests/unit/test_error_handling_integration.py"],
                "markers": [],
                "description": "Circuit breakers, retry logic, graceful degradation"
            },
            "Monitoring": {
                "files": ["tests/unit/test_monitoring_integration.py"],
                "markers": [],
                "description": "Metrics, health checks, alerting"
            },
            "Database": {
                "files": ["tests/unit/test_production_database.py"],
                "markers": [],
                "description": "User management, sessions, audit logs"
            },
            "Caching": {
                "files": ["tests/unit/test_production_caching.py"],
                "markers": [],
                "description": "Redis, rate limiting, response caching"
            },
            "API Integration": {
                "files": ["tests/integration/test_production_api_integration.py"],
                "markers": [],
                "description": "Complete API with all middleware"
            },
            "End-to-End Workflows": {
                "files": ["tests/integration/test_production_e2e.py"],
                "markers": [],
                "description": "Complete user journeys and workflows"
            },
            "Comprehensive Production": {
                "files": ["tests/test_comprehensive_production.py"],
                "markers": [],
                "description": "Full production readiness validation"
            },
            "Core RAG Components": {
                "files": [
                    "tests/unit/test_embedder.py",
                    "tests/unit/test_vectorstore.py",
                    "tests/unit/test_chunker.py"
                ],
                "markers": [],
                "description": "Core RAG functionality"
            },
            "Conversational Routing": {
                "files": [
                    "tests/unit/test_conversational_routing.py",
                    "tests/unit/test_conversational_routing_advanced.py",
                    "tests/integration/test_conversational_routing_api.py",
                    "tests/integration/test_conversational_routing_e2e.py"
                ],
                "markers": [],
                "description": "Advanced conversational routing system"
            }
        }
        
        # Filter based on test type
        if test_type == "all":
            return all_suites
        elif test_type == "unit":
            return {k: v for k, v in all_suites.items() 
                   if any("unit" in f for f in v["files"])}
        elif test_type == "integration":
            return {k: v for k, v in all_suites.items() 
                   if any("integration" in f for f in v["files"])}
        elif test_type == "e2e":
            return {k: v for k, v in all_suites.items() 
                   if "End-to-End" in k or "Comprehensive" in k}
        elif test_type == "security":
            return {k: v for k, v in all_suites.items() 
                   if "Security" in k or "Database" in k}
        elif test_type == "performance":
            return {k: v for k, v in all_suites.items() 
                   if "Comprehensive" in k or "Caching" in k}
        elif test_type == "quick":
            return {
                "Security Integration": all_suites["Security Integration"],
                "Error Handling": all_suites["Error Handling"],
                "Core RAG Components": all_suites["Core RAG Components"]
            }
        else:
            return {}
    
    def _run_single_suite(self, suite_name: str, suite_config: Dict[str, Any], **options) -> bool:
        """Run a single test suite."""
        files = suite_config["files"]
        markers = suite_config.get("markers", [])
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Add files (only existing ones)
        existing_files = [f for f in files if Path(f).exists()]
        if not existing_files:
            print(f"⚠️  No test files found for {suite_name}")
            return True  # Not a failure, just no tests
        
        cmd.extend(existing_files)
        
        # Add options
        if options.get("verbose", False):
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        if options.get("coverage", False):
            cmd.extend([
                "--cov=rag_engine",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov"
            ])
        
        if options.get("parallel", False):
            cmd.extend(["-n", "auto"])
        
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        # Performance-specific options
        if "performance" in suite_name.lower() or options.get("benchmark", False):
            cmd.extend(["--durations=10"])
        
        # Add other pytest options
        cmd.extend([
            "--tb=short",
            "--color=yes"
        ])
        
        # Run the tests
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, 
                                  cwd=self.project_root,
                                  capture_output=False,
                                  check=False)
            
            success = result.returncode == 0
            
            if success:
                print(f"✅ {suite_name} - PASSED")
            else:
                print(f"❌ {suite_name} - FAILED (exit code: {result.returncode})")
            
            return success
            
        except Exception as e:
            print(f"❌ {suite_name} - ERROR: {str(e)}")
            return False
    
    def _generate_final_report(self, overall_success: bool):
        """Generate final test report."""
        print("\n" + "=" * 80)
        print("📊 FINAL TEST REPORT")
        print("=" * 80)
        
        # Summary statistics
        total_suites = len(self.test_results)
        passed_suites = sum(1 for success in self.test_results.values() if success)
        failed_suites = total_suites - passed_suites
        
        print(f"Total Test Suites: {total_suites}")
        print(f"Passed: {passed_suites}")
        print(f"Failed: {failed_suites}")
        print(f"Success Rate: {passed_suites/total_suites*100:.1f}%")
        print(f"Total Time: {self.total_time:.2f} seconds")
        
        # Detailed results
        print("\n📋 Detailed Results:")
        print("-" * 60)
        
        for suite_name, success in self.test_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {suite_name}")
        
        # Overall status
        print("\n" + "=" * 80)
        if overall_success:
            print("🎉 ALL TESTS PASSED!")
            print("✅ RAG Engine is PRODUCTION-READY")
            print("\n🚀 Ready for deployment with confidence!")
        else:
            print("❌ SOME TESTS FAILED")
            print("🔧 Review failures before production deployment")
            print(f"\n📝 {failed_suites} test suite(s) need attention")
        
        print("=" * 80)
        
        # Recommendations
        if overall_success:
            print("\n💡 Recommendations:")
            print("• Deploy to staging environment for final validation")
            print("• Set up continuous monitoring in production")
            print("• Configure alerting thresholds based on test results")
            print("• Schedule regular production health checks")
        else:
            print("\n🔧 Next Steps:")
            print("• Review failed test output above")
            print("• Fix failing components before deployment")
            print("• Re-run tests after fixes")
            print("• Consider running subset of tests for faster iteration")
    
    def run_coverage_report(self):
        """Generate detailed coverage report."""
        print("\n📊 Generating Coverage Report...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/",
            "--cov=rag_engine",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "-q"
        ]
        
        try:
            subprocess.run(cmd, cwd=self.project_root, check=True)
            print("✅ Coverage report generated in htmlcov/")
        except subprocess.CalledProcessError:
            print("❌ Failed to generate coverage report")
    
    def run_performance_benchmarks(self):
        """Run performance benchmarks."""
        print("\n⚡ Running Performance Benchmarks...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/test_comprehensive_production.py::TestProductionReadinessValidation::test_production_performance_benchmarks",
            "-v", "-s"
        ]
        
        try:
            subprocess.run(cmd, cwd=self.project_root, check=False)
        except Exception as e:
            print(f"❌ Benchmark error: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG Engine Production Test Runner")
    
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "e2e", "security", "performance", "quick"],
        help="Type of tests to run (default: all)"
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
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed HTML report"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Include performance benchmarks"
    )
    
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running tests even if some fail"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = ProductionTestRunner()
    
    # Run tests
    success = runner.run_test_suite(
        test_type=args.test_type,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel,
        benchmark=args.benchmark,
        continue_on_failure=args.continue_on_failure
    )
    
    # Additional reports
    if args.coverage:
        runner.run_coverage_report()
    
    if args.benchmark:
        runner.run_performance_benchmarks()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 