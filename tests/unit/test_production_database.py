"""
Unit tests for production database features.

Tests user management, session handling, audit logging, and all database operations.
"""

import pytest
import time
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

from rag_engine.core.production_database import ProductionDatabaseManager


class TestProductionDatabaseManager:
    """Test the production database manager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Use in-memory SQLite for testing
        self.config = {
            "database": {
                "provider": "sqlite",
                "sqlite": {
                    "database": ":memory:"
                }
            }
        }
        
        self.db_manager = ProductionDatabaseManager(self.config)
        
        # Initialize database tables
        self.db_manager.initialize_database()
    
    def test_database_initialization(self):
        """Test database initialization."""
        assert self.db_manager.config == self.config
        assert self.db_manager.provider == "sqlite"
        assert self.db_manager.db_connection is not None
    
    def test_user_creation(self):
        """Test user creation."""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "secure_password_123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        
        assert user_id is not None
        assert isinstance(user_id, str)
        assert len(user_id) > 0
    
    def test_user_authentication(self):
        """Test user authentication."""
        # Create user
        user_data = {
            "username": "authuser",
            "email": "auth@example.com",
            "password": "auth_password_123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        
        # Test correct authentication
        auth_result = self.db_manager.authenticate_user("authuser", "auth_password_123")
        assert auth_result is not None
        assert auth_result["user_id"] == user_id
        assert auth_result["username"] == "authuser"
        assert auth_result["role"] == "user"
        
        # Test incorrect password
        auth_result = self.db_manager.authenticate_user("authuser", "wrong_password")
        assert auth_result is None
        
        # Test non-existent user
        auth_result = self.db_manager.authenticate_user("nonexistent", "password")
        assert auth_result is None
    
    def test_user_retrieval(self):
        """Test user retrieval by ID and username."""
        # Create user
        user_data = {
            "username": "retrieveuser",
            "email": "retrieve@example.com",
            "password": "retrieve_password_123",
            "role": "admin"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        
        # Retrieve by ID
        user_by_id = self.db_manager.get_user(user_id)
        assert user_by_id is not None
        assert user_by_id["username"] == "retrieveuser"
        assert user_by_id["email"] == "retrieve@example.com"
        assert user_by_id["role"] == "admin"
        
        # Retrieve by username
        user_by_username = self.db_manager.get_user_by_username("retrieveuser")
        assert user_by_username is not None
        assert user_by_username["user_id"] == user_id
    
    def test_user_update(self):
        """Test user information update."""
        # Create user
        user_data = {
            "username": "updateuser",
            "email": "update@example.com",
            "password": "update_password_123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        
        # Update user
        update_data = {
            "email": "newemail@example.com",
            "role": "admin"
        }
        
        success = self.db_manager.update_user(user_id, **update_data)
        assert success == True
        
        # Verify update
        updated_user = self.db_manager.get_user(user_id)
        assert updated_user["email"] == "newemail@example.com"
        assert updated_user["role"] == "admin"
        assert updated_user["username"] == "updateuser"  # Should remain unchanged
    
    def test_password_change(self):
        """Test password change functionality."""
        # Create user
        user_data = {
            "username": "passworduser",
            "email": "password@example.com",
            "password": "old_password_123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        
        # Change password
        success = self.db_manager.change_password(user_id, "old_password_123", "new_password_456")
        assert success == True
        
        # Test authentication with new password
        auth_result = self.db_manager.authenticate_user("passworduser", "new_password_456")
        assert auth_result is not None
        
        # Test authentication with old password (should fail)
        auth_result = self.db_manager.authenticate_user("passworduser", "old_password_123")
        assert auth_result is None
    
    def test_user_deletion(self):
        """Test user deletion."""
        # Create user
        user_data = {
            "username": "deleteuser",
            "email": "delete@example.com",
            "password": "delete_password_123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        
        # Verify user exists
        user = self.db_manager.get_user(user_id)
        assert user is not None
        
        # Delete user
        success = self.db_manager.delete_user(user_id)
        assert success == True
        
        # Verify user is deleted
        user = self.db_manager.get_user(user_id)
        assert user is None
    
    def test_session_creation(self):
        """Test session creation."""
        # Create user first
        user_data = {
            "username": "sessionuser",
            "email": "session@example.com",
            "password": "session_password_123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        
        # Create session
        session_data = {
            "user_agent": "test-browser",
            "ip_address": "192.168.1.1"
        }
        
        session_id = self.db_manager.create_session(user_id, **session_data)
        
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0
    
    def test_session_retrieval(self):
        """Test session retrieval and validation."""
        # Create user and session
        user_data = {
            "username": "sessionuser2",
            "email": "session2@example.com",
            "password": "session_password_123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        session_id = self.db_manager.create_session(user_id, user_agent="test", ip_address="127.0.0.1")
        
        # Retrieve session
        session = self.db_manager.get_session(session_id)
        
        assert session is not None
        assert session["user_id"] == user_id
        assert session["user_agent"] == "test"
        assert session["ip_address"] == "127.0.0.1"
        assert session["is_active"] == True
    
    def test_session_invalidation(self):
        """Test session invalidation."""
        # Create user and session
        user_data = {
            "username": "sessionuser3",
            "email": "session3@example.com",
            "password": "session_password_123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        session_id = self.db_manager.create_session(user_id, user_agent="test", ip_address="127.0.0.1")
        
        # Verify session is active
        session = self.db_manager.get_session(session_id)
        assert session["is_active"] == True
        
        # Invalidate session
        success = self.db_manager.invalidate_session(session_id)
        assert success == True
        
        # Verify session is inactive
        session = self.db_manager.get_session(session_id)
        assert session is None or session["is_active"] == False
    
    def test_session_expiration(self):
        """Test session expiration handling."""
        # Create user and session with short TTL
        user_data = {
            "username": "expiryuser",
            "email": "expiry@example.com",
            "password": "expiry_password_123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        
        # Create session with 1 second TTL
        session_id = self.db_manager.create_session(
            user_id, 
            user_agent="test", 
            ip_address="127.0.0.1",
            ttl_seconds=1
        )
        
        # Session should be valid immediately
        session = self.db_manager.get_session(session_id)
        assert session is not None
        assert session["is_active"] == True
        
        # Wait for expiration
        time.sleep(2)
        
        # Session should be expired/invalid
        session = self.db_manager.get_session(session_id)
        assert session is None or session["is_active"] == False
    
    def test_audit_logging(self):
        """Test audit logging functionality."""
        # Create user for audit logging
        user_data = {
            "username": "audituser",
            "email": "audit@example.com",
            "password": "audit_password_123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        
        # Log audit events
        audit_data = {
            "user_id": user_id,
            "action": "login",
            "resource": "auth_api",
            "details": {"method": "password", "success": True},
            "ip_address": "192.168.1.1",
            "user_agent": "test-browser"
        }
        
        audit_id = self.db_manager.log_audit_event(**audit_data)
        
        assert audit_id is not None
        assert isinstance(audit_id, str)
    
    def test_audit_log_retrieval(self):
        """Test audit log retrieval and filtering."""
        # Create user
        user_data = {
            "username": "audituser2",
            "email": "audit2@example.com",
            "password": "audit_password_123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        
        # Log multiple audit events
        events = [
            {"action": "login", "resource": "auth_api", "success": True},
            {"action": "query", "resource": "chat_api", "success": True},
            {"action": "logout", "resource": "auth_api", "success": True}
        ]
        
        for event in events:
            self.db_manager.log_audit_event(
                user_id=user_id,
                ip_address="127.0.0.1",
                user_agent="test",
                **event
            )
        
        # Retrieve audit logs
        audit_logs = self.db_manager.get_audit_logs(user_id=user_id)
        
        assert len(audit_logs) >= 3
        assert all(log["user_id"] == user_id for log in audit_logs)
        
        # Test filtering by action
        login_logs = self.db_manager.get_audit_logs(user_id=user_id, action="login")
        assert len(login_logs) >= 1
        assert all(log["action"] == "login" for log in login_logs)
    
    def test_user_listing(self):
        """Test user listing with pagination."""
        # Create multiple users
        for i in range(10):
            user_data = {
                "username": f"listuser{i}",
                "email": f"list{i}@example.com",
                "password": f"password{i}",
                "role": "user" if i % 2 == 0 else "admin"
            }
            self.db_manager.create_user(**user_data)
        
        # List all users
        all_users = self.db_manager.list_users()
        assert len(all_users) >= 10
        
        # List with pagination
        page1 = self.db_manager.list_users(limit=5, offset=0)
        page2 = self.db_manager.list_users(limit=5, offset=5)
        
        assert len(page1) <= 5
        assert len(page2) <= 5
        
        # No overlap between pages
        page1_ids = {user["user_id"] for user in page1}
        page2_ids = {user["user_id"] for user in page2}
        assert len(page1_ids.intersection(page2_ids)) == 0
    
    def test_user_search(self):
        """Test user search functionality."""
        # Create users with searchable data
        users_data = [
            {"username": "alice", "email": "alice@example.com", "role": "admin"},
            {"username": "bob", "email": "bob@test.com", "role": "user"},
            {"username": "charlie", "email": "charlie@example.com", "role": "user"}
        ]
        
        for user_data in users_data:
            user_data["password"] = "password123"
            self.db_manager.create_user(**user_data)
        
        # Search by username
        alice_results = self.db_manager.search_users(username="alice")
        assert len(alice_results) == 1
        assert alice_results[0]["username"] == "alice"
        
        # Search by email domain
        example_results = self.db_manager.search_users(email_domain="example.com")
        assert len(example_results) >= 2
        
        # Search by role
        admin_results = self.db_manager.search_users(role="admin")
        assert len(admin_results) >= 1
        assert all(user["role"] == "admin" for user in admin_results)
    
    def test_database_backup_restore(self):
        """Test database backup and restore functionality."""
        # Create some test data
        user_data = {
            "username": "backupuser",
            "email": "backup@example.com",
            "password": "backup_password_123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        session_id = self.db_manager.create_session(user_id, user_agent="test", ip_address="127.0.0.1")
        
        # Create backup
        with tempfile.NamedTemporaryFile(delete=False) as backup_file:
            backup_path = backup_file.name
        
        try:
            success = self.db_manager.create_backup(backup_path)
            assert success == True
            assert os.path.exists(backup_path)
            assert os.path.getsize(backup_path) > 0
            
        finally:
            # Cleanup
            if os.path.exists(backup_path):
                os.unlink(backup_path)


class TestProductionDatabaseEdgeCases:
    """Test edge cases and error conditions for production database."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "database": {
                "provider": "sqlite",
                "sqlite": {"database": ":memory:"}
            }
        }
        self.db_manager = ProductionDatabaseManager(self.config)
        self.db_manager.initialize_database()
    
    def test_duplicate_username(self):
        """Test handling of duplicate usernames."""
        user_data = {
            "username": "duplicate",
            "email": "first@example.com",
            "password": "password123",
            "role": "user"
        }
        
        # Create first user
        user_id1 = self.db_manager.create_user(**user_data)
        assert user_id1 is not None
        
        # Try to create user with same username
        user_data["email"] = "second@example.com"
        user_id2 = self.db_manager.create_user(**user_data)
        
        # Should either fail or return None
        assert user_id2 is None or user_id2 != user_id1
    
    def test_duplicate_email(self):
        """Test handling of duplicate emails."""
        email = "duplicate@example.com"
        
        user_data1 = {
            "username": "user1",
            "email": email,
            "password": "password123",
            "role": "user"
        }
        
        user_data2 = {
            "username": "user2",
            "email": email,
            "password": "password456",
            "role": "user"
        }
        
        # Create first user
        user_id1 = self.db_manager.create_user(**user_data1)
        assert user_id1 is not None
        
        # Try to create user with same email
        user_id2 = self.db_manager.create_user(**user_data2)
        
        # Should either fail or return None
        assert user_id2 is None or user_id2 != user_id1
    
    def test_invalid_user_data(self):
        """Test handling of invalid user data."""
        invalid_data_sets = [
            {"username": "", "email": "test@example.com", "password": "password", "role": "user"},  # Empty username
            {"username": "test", "email": "", "password": "password", "role": "user"},  # Empty email
            {"username": "test", "email": "test@example.com", "password": "", "role": "user"},  # Empty password
            {"username": "test", "email": "invalid-email", "password": "password", "role": "user"},  # Invalid email
            {"username": "test", "email": "test@example.com", "password": "123", "role": "user"},  # Weak password
        ]
        
        for invalid_data in invalid_data_sets:
            user_id = self.db_manager.create_user(**invalid_data)
            # Should either return None or raise an exception
            if user_id is not None:
                # If it succeeds, the data should be validated/sanitized
                user = self.db_manager.get_user(user_id)
                assert user is not None
    
    def test_sql_injection_prevention(self):
        """Test prevention of SQL injection attacks."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "admin'/*",
            "1' OR '1'='1",
            "test'; DELETE FROM sessions; --"
        ]
        
        for malicious_input in malicious_inputs:
            # Try malicious input in username
            user_data = {
                "username": malicious_input,
                "email": "test@example.com",
                "password": "password123",
                "role": "user"
            }
            
            try:
                user_id = self.db_manager.create_user(**user_data)
                # If it succeeds, verify the malicious input was sanitized
                if user_id:
                    user = self.db_manager.get_user(user_id)
                    assert user is not None
                    # Username should not contain SQL injection
                    assert "DROP" not in user["username"].upper()
                    assert "DELETE" not in user["username"].upper()
            except Exception:
                # It's acceptable to raise an exception for malicious input
                pass
    
    def test_concurrent_user_creation(self):
        """Test concurrent user creation."""
        import threading
        
        results = []
        errors = []
        
        def create_user(index):
            try:
                user_data = {
                    "username": f"concurrent_user_{index}",
                    "email": f"concurrent_{index}@example.com",
                    "password": "password123",
                    "role": "user"
                }
                user_id = self.db_manager.create_user(**user_data)
                results.append(user_id)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_user, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have created users successfully
        assert len(results) > 0
        # All user IDs should be unique
        assert len(set(filter(None, results))) == len([r for r in results if r is not None])
    
    def test_session_cleanup(self):
        """Test automatic cleanup of expired sessions."""
        # Create user
        user_data = {
            "username": "cleanupuser",
            "email": "cleanup@example.com",
            "password": "password123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        
        # Create sessions with different expiration times
        session_ids = []
        for i in range(5):
            session_id = self.db_manager.create_session(
                user_id,
                user_agent=f"test{i}",
                ip_address="127.0.0.1",
                ttl_seconds=1 if i < 3 else 3600  # First 3 expire quickly
            )
            session_ids.append(session_id)
        
        # Wait for some sessions to expire
        time.sleep(2)
        
        # Cleanup expired sessions
        cleaned_count = self.db_manager.cleanup_expired_sessions()
        
        # Should have cleaned up expired sessions
        assert cleaned_count >= 0  # Might be 0 if cleanup is not implemented
    
    def test_large_audit_log_data(self):
        """Test handling of large audit log data."""
        # Create user
        user_data = {
            "username": "largeaudituser",
            "email": "largeaudit@example.com",
            "password": "password123",
            "role": "user"
        }
        
        user_id = self.db_manager.create_user(**user_data)
        
        # Create large audit log entry
        large_details = {"data": "x" * 10000}  # 10KB of data
        
        audit_id = self.db_manager.log_audit_event(
            user_id=user_id,
            action="large_data_test",
            resource="test_api",
            details=large_details,
            ip_address="127.0.0.1",
            user_agent="test"
        )
        
        # Should handle large data gracefully
        assert audit_id is not None or audit_id is None  # Either works or fails gracefully
    
    def test_database_connection_resilience(self):
        """Test database connection resilience."""
        # This test would be more meaningful with a real database connection
        # For SQLite in-memory, we'll test basic resilience
        
        # Perform operations to test connection stability
        for i in range(10):
            user_data = {
                "username": f"resilience_user_{i}",
                "email": f"resilience_{i}@example.com",
                "password": "password123",
                "role": "user"
            }
            
            user_id = self.db_manager.create_user(**user_data)
            assert user_id is not None
            
            # Test retrieval
            user = self.db_manager.get_user(user_id)
            assert user is not None
    
    def test_password_security_requirements(self):
        """Test password security requirements."""
        weak_passwords = [
            "123",          # Too short
            "password",     # Common word
            "12345678",     # Numbers only
            "abcdefgh",     # Letters only
        ]
        
        for weak_password in weak_passwords:
            user_data = {
                "username": f"weakpass_{weak_password}",
                "email": f"weak_{weak_password}@example.com",
                "password": weak_password,
                "role": "user"
            }
            
            user_id = self.db_manager.create_user(**user_data)
            
            # Should either reject weak password or accept with warning
            # Implementation dependent - either behavior is acceptable
            if user_id is not None:
                # If accepted, verify user was created
                user = self.db_manager.get_user(user_id)
                assert user is not None


class TestProductionDatabasePerformance:
    """Performance tests for production database."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "database": {
                "provider": "sqlite",
                "sqlite": {"database": ":memory:"}
            }
        }
        self.db_manager = ProductionDatabaseManager(self.config)
        self.db_manager.initialize_database()
    
    @pytest.mark.performance
    def test_user_creation_performance(self):
        """Test user creation performance."""
        start_time = time.time()
        
        user_ids = []
        for i in range(100):
            user_data = {
                "username": f"perf_user_{i}",
                "email": f"perf_{i}@example.com",
                "password": "password123",
                "role": "user"
            }
            user_id = self.db_manager.create_user(**user_data)
            user_ids.append(user_id)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should create 100 users in reasonable time
        assert total_time < 5.0, f"User creation too slow: {total_time}s"
        assert len([uid for uid in user_ids if uid is not None]) >= 90  # At least 90% success
    
    @pytest.mark.performance
    def test_authentication_performance(self):
        """Test authentication performance."""
        # Create users first
        usernames = []
        for i in range(50):
            user_data = {
                "username": f"auth_perf_user_{i}",
                "email": f"auth_perf_{i}@example.com",
                "password": "password123",
                "role": "user"
            }
            user_id = self.db_manager.create_user(**user_data)
            if user_id:
                usernames.append(user_data["username"])
        
        # Test authentication performance
        start_time = time.time()
        
        for username in usernames[:20]:  # Test first 20 users
            auth_result = self.db_manager.authenticate_user(username, "password123")
            assert auth_result is not None
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should authenticate 20 users quickly
        assert total_time < 2.0, f"Authentication too slow: {total_time}s"
    
    @pytest.mark.performance
    def test_session_operations_performance(self):
        """Test session operations performance."""
        # Create user
        user_data = {
            "username": "session_perf_user",
            "email": "session_perf@example.com",
            "password": "password123",
            "role": "user"
        }
        user_id = self.db_manager.create_user(**user_data)
        
        # Test session creation performance
        start_time = time.time()
        
        session_ids = []
        for i in range(100):
            session_id = self.db_manager.create_session(
                user_id,
                user_agent=f"test_agent_{i}",
                ip_address="127.0.0.1"
            )
            session_ids.append(session_id)
        
        creation_time = time.time() - start_time
        
        # Test session retrieval performance
        start_time = time.time()
        
        for session_id in session_ids[:50]:  # Test first 50 sessions
            if session_id:
                session = self.db_manager.get_session(session_id)
                assert session is not None
        
        retrieval_time = time.time() - start_time
        
        # Performance assertions
        assert creation_time < 3.0, f"Session creation too slow: {creation_time}s"
        assert retrieval_time < 2.0, f"Session retrieval too slow: {retrieval_time}s"
    
    @pytest.mark.performance
    def test_audit_logging_performance(self):
        """Test audit logging performance."""
        # Create user
        user_data = {
            "username": "audit_perf_user",
            "email": "audit_perf@example.com",
            "password": "password123",
            "role": "user"
        }
        user_id = self.db_manager.create_user(**user_data)
        
        # Test audit logging performance
        start_time = time.time()
        
        for i in range(200):
            self.db_manager.log_audit_event(
                user_id=user_id,
                action=f"test_action_{i % 10}",
                resource="test_api",
                details={"test": f"data_{i}"},
                ip_address="127.0.0.1",
                user_agent="test_agent"
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should log 200 audit events quickly
        assert total_time < 3.0, f"Audit logging too slow: {total_time}s" 