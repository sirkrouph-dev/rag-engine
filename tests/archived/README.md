# Archived Test Files

This directory contains test files from the development process that have been superseded by the comprehensive test suite.

## 📁 Contents

### **Development Test Files**
These are individual test files created during the development process:
- `test_*.py` - Various component and integration tests
- `test_config*.json` - Test configuration files
- `test_doc.txt` - Test document

### **Legacy Scripts**
- `debug_base_api.py` - Old debugging script
- `demo_orchestration.py` - Early orchestration demo
- `run_tests.py` - Old test runner script

## 🎯 **Current Status**

These files are **archived** and no longer actively used. The current testing approach uses:

1. **`tests/test_comprehensive.py`** - Main comprehensive test suite
2. **`tests/unit/`** - Organized unit tests
3. **`tests/integration/`** - Organized integration tests

## 🔧 **Why Archived?**

During development, we created many individual test files to validate specific components and features. As the project matured, we:

1. **Consolidated testing** into a comprehensive test suite
2. **Organized tests** into unit and integration categories
3. **Reduced duplication** by removing redundant test files
4. **Improved maintainability** with a cleaner test structure

## ⚠️ **Note**

These files are kept for reference but are not maintained or guaranteed to work with the current codebase. For current testing, use the main test suite in the parent directory.
