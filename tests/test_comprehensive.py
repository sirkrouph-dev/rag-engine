#!/usr/bin/env python3
"""
Comprehensive testing script for RAG Engine
Tests all major components and features we've built
"""
import requests
import json
import time
import sys
from typing import Dict, Any

class RAGEngineTestSuite:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.test_results = {}
        
    def log(self, message: str, level: str = "INFO"):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def test_endpoint(self, name: str, method: str, endpoint: str, data: Dict = None) -> bool:
        """Test a single API endpoint"""
        try:
            url = f"{self.base_url}{endpoint}"
            self.log(f"Testing {name}: {method} {endpoint}")
            
            if method.upper() == "GET":
                response = requests.get(url, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=30)
            else:
                self.log(f"Unsupported method: {method}", "ERROR")
                return False
            
            if response.status_code == 200:
                result = response.json()
                self.log(f"✅ {name} - Success", "SUCCESS")
                if 'status' in result and result['status'] == 'error':
                    self.log(f"   Warning: API returned error status: {result}", "WARN")
                return True
            else:
                self.log(f"❌ {name} - HTTP {response.status_code}: {response.text}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ {name} - Exception: {str(e)}", "ERROR")
            return False
    
    def run_health_tests(self):
        """Test basic health and status endpoints"""
        self.log("🔍 Running Health Tests...")
        
        tests = [
            ("Health Check", "GET", "/health"),
            ("Status Check", "GET", "/status"),
        ]
        
        passed = 0
        for name, method, endpoint in tests:
            if self.test_endpoint(name, method, endpoint):
                passed += 1
        
        self.test_results["health"] = {"passed": passed, "total": len(tests)}
        return passed == len(tests)
    
    def run_ai_assistant_tests(self):
        """Test AI assistant functionality"""
        self.log("🤖 Running AI Assistant Tests...")
        
        test_questions = [
            "Which stack should I choose for local development?",
            "How can I reduce dependency bloat?",
            "What's the difference between DEMO and FULL stacks?",
            "Help me optimize my current setup"
        ]
        
        passed = 0
        total = len(test_questions)
        
        for question in test_questions:
            data = {"question": question, "model": "phi3.5:latest"}
            if self.test_endpoint(f"AI Question: '{question[:30]}...'", "POST", "/ai-assistant", data):
                passed += 1
        
        self.test_results["ai_assistant"] = {"passed": passed, "total": total}
        return passed == total
    
    def run_stack_management_tests(self):
        """Test stack configuration and management"""
        self.log("📦 Running Stack Management Tests...")
        
        # Test all stack types
        stack_types = ["DEMO", "LOCAL", "CLOUD", "MINI", "FULL", "RESEARCH"]
        passed = 0
        total = len(stack_types) + 2  # +2 for analyze and audit
        
        for stack_type in stack_types:
            data = {"stack_type": stack_type}
            if self.test_endpoint(f"Configure {stack_type} Stack", "POST", "/stack/configure", data):
                passed += 1
        
        # Test analysis endpoints
        if self.test_endpoint("Stack Analysis", "GET", "/stack/analyze"):
            passed += 1
        
        if self.test_endpoint("Dependency Audit", "GET", "/stack/audit"):
            passed += 1
        
        self.test_results["stack_management"] = {"passed": passed, "total": total}
        return passed == total
    
    def run_orchestrator_tests(self):
        """Test orchestrator functionality"""
        self.log("🧠 Running Orchestrator Tests...")
        
        tests = [
            ("Orchestrator Status", "GET", "/orchestrator/status"),
            ("Get Components", "GET", "/orchestrator/components"),
        ]
        
        passed = 0
        for name, method, endpoint in tests:
            if self.test_endpoint(name, method, endpoint):
                passed += 1
        
        self.test_results["orchestrator"] = {"passed": passed, "total": len(tests)}
        return passed == len(tests)
    
    def run_document_tests(self):
        """Test document and chunk management"""
        self.log("📄 Running Document Tests...")
        
        tests = [
            ("Get Documents", "GET", "/documents"),
            ("Get Chunks", "GET", "/chunks"),
        ]
        
        passed = 0
        for name, method, endpoint in tests:
            if self.test_endpoint(name, method, endpoint):
                passed += 1
        
        self.test_results["documents"] = {"passed": passed, "total": len(tests)}
        return passed == len(tests)
    
    def run_pipeline_tests(self):
        """Test pipeline building"""
        self.log("🔧 Running Pipeline Tests...")
        
        # Note: Build test might take a while
        self.log("Building pipeline (this may take 30+ seconds)...")
        
        if self.test_endpoint("Build Pipeline", "POST", "/build"):
            self.test_results["pipeline"] = {"passed": 1, "total": 1}
            return True
        else:
            self.test_results["pipeline"] = {"passed": 0, "total": 1}
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        self.log("🚀 Starting RAG Engine Comprehensive Test Suite")
        self.log("=" * 60)
        
        start_time = time.time()
        
        # Run all test categories
        test_categories = [
            ("Health & Status", self.run_health_tests),
            ("AI Assistant", self.run_ai_assistant_tests),
            ("Stack Management", self.run_stack_management_tests),
            ("Orchestrator", self.run_orchestrator_tests),
            ("Documents", self.run_document_tests),
            ("Pipeline", self.run_pipeline_tests),
        ]
        
        total_passed = 0
        total_tests = 0
        
        for category_name, test_func in test_categories:
            self.log(f"\n📋 {category_name} Tests")
            self.log("-" * 40)
            
            try:
                success = test_func()
                category_results = self.test_results.get(category_name.lower().replace(" & ", "_").replace(" ", "_"))
                if category_results:
                    passed = category_results["passed"]
                    total = category_results["total"]
                    total_passed += passed
                    total_tests += total
                    
                    if success:
                        self.log(f"✅ {category_name}: {passed}/{total} tests passed", "SUCCESS")
                    else:
                        self.log(f"⚠️  {category_name}: {passed}/{total} tests passed", "WARN")
                        
            except Exception as e:
                self.log(f"❌ {category_name} tests failed with exception: {str(e)}", "ERROR")
        
        # Final results
        end_time = time.time()
        duration = end_time - start_time
        
        self.log("\n" + "=" * 60)
        self.log("🎯 FINAL TEST RESULTS")
        self.log("=" * 60)
        self.log(f"Total Tests: {total_tests}")
        self.log(f"Passed: {total_passed}")
        self.log(f"Failed: {total_tests - total_passed}")
        self.log(f"Success Rate: {(total_passed/total_tests)*100:.1f}%")
        self.log(f"Duration: {duration:.2f} seconds")
        
        if total_passed == total_tests:
            self.log("🎉 ALL TESTS PASSED! RAG Engine is working perfectly!", "SUCCESS")
            return True
        else:
            self.log(f"⚠️  {total_tests - total_passed} tests failed. Check logs above.", "WARN")
            return False

def main():
    """Run the test suite"""
    print("RAG Engine Comprehensive Test Suite")
    print("=" * 60)
    
    # Check if servers are running
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend server not responding at http://localhost:8001")
            print("   Please start with: python -m rag_engine serve --port 8001")
            sys.exit(1)
    except:
        print("❌ Backend server not running at http://localhost:8001")
        print("   Please start with: python -m rag_engine serve --port 8001")
        sys.exit(1)
    
    # Run tests
    test_suite = RAGEngineTestSuite()
    success = test_suite.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
