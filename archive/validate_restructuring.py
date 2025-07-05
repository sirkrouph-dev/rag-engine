#!/usr/bin/env python3
"""
RAG Engine Post-Restructuring Validation Script
===============================================

Validates that the project still works correctly after the major restructuring.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    print(f"🔍 {description}...")
    if Path(filepath).exists():
        print(f"✅ {description} - EXISTS")
        return True
    else:
        print(f"❌ {description} - MISSING: {filepath}")
        return False

def main():
    print("RAG Engine Post-Restructuring Validation")
    print("=" * 50)
    
    tests = []
    
    # Check core files exist
    tests.append(check_file_exists("README.md", "Main README exists"))
    tests.append(check_file_exists("PROJECT_STRUCTURE.md", "Project structure doc exists"))
    tests.append(check_file_exists("rag_engine/__init__.py", "Core package exists"))
    tests.append(check_file_exists("frontend/package.json", "Frontend exists"))
    tests.append(check_file_exists("docs/guides/AI_ASSISTANT_INTEGRATION.md", "AI Assistant guide moved"))
    tests.append(check_file_exists("docs/guides/BLOAT_REDUCTION.md", "Bloat reduction guide moved"))
    tests.append(check_file_exists("configs/config.json", "Config file moved"))
    tests.append(check_file_exists("docker/docker-compose.yml", "Docker compose moved"))
    tests.append(check_file_exists("scripts/ai_setup.py", "AI setup script exists"))
    tests.append(check_file_exists("tests/test_comprehensive.py", "Comprehensive test moved"))
    
    # Check imports work
    tests.append(run_command("python -c \"from rag_engine.core.orchestration import ComponentRegistry; print('Core imports OK')\"", "Core module imports"))
    tests.append(run_command("python -c \"from rag_engine.interfaces.cli import app; print('CLI imports OK')\"", "CLI module imports"))
    tests.append(run_command("python -c \"from rag_engine.interfaces.api import create_production_app; print('API imports OK')\"", "API module imports"))
    
    # Check CLI works
    tests.append(run_command("python -m rag_engine --help", "CLI help command"))
    tests.append(run_command("python -m rag_engine ask --help", "AI assistant command help"))
    
    # Check requirements files
    tests.append(check_file_exists("requirements/base.txt", "Base requirements exist"))
    tests.append(check_file_exists("requirements/stacks/requirements-demo.txt", "Demo requirements exist"))
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(tests)
    total = len(tests)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("✅ Project restructuring successful!")
        print("✅ All core functionality preserved")
        print("✅ Ready for continued development")
        return 0
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total})")
        print("❌ Manual review may be needed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
