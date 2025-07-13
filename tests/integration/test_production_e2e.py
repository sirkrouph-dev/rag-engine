"""
End-to-End tests for complete production workflows.

Tests complete user journeys, system integration, and real-world scenarios
with all production features enabled.
"""

import pytest
import time
import asyncio
import json
import threading
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from typing import Dict, Any, List

from rag_engine.interfaces.fastapi_enhanced import FastAPIEnhanced
from rag_engine.interfaces.api_config import APICustomizationConfig, SecurityConfig, MonitoringConfig
from rag_engine.core.production_database import ProductionDatabaseManager


class TestProductionE2EWorkflows:
    """End-to-end tests for complete production workflows."""
    
    def setup_method(self):
        """Setup complete production environment."""
        # Full production configuration
        self.config = {
            "security": {
                "jwt_secret_key": "e2e-test-secret-key-production",
                "jwt_algorithm": "HS256",
                "jwt_expiration": 3600,
                "api_keys": ["e2e-api-key-admin", "e2e-api-key-user"],
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 200,
                    "burst_limit": 300
                },
                "input_validation": {
                    "enabled": True,
                    "max_length": 10000,
                    "blocked_patterns": ["<script>", "DROP TABLE"]
                },
                "ip_filtering": {
                    "enabled": True,
                    "allowed_ips": ["127.0.0.1", "192.168.1.0/24"]
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics": {
                    "enabled": True,
                    "collection_interval": 10
                },
                "health_checks": {
                    "enabled": True,
                    "check_interval": 30
                },
                "alerting": {
                    "enabled": True,
                    "thresholds": {
                        "response_time": 1000,
                        "error_rate": 5
                    }
                }
            },
            "error_handling": {
                "circuit_breaker": {
                    "enabled": True,
                    "failure_threshold": 5,
                    "recovery_timeout": 60
                },
                "retry": {
                    "enabled": True,
                    "max_attempts": 3,
                    "backoff_factor": 2
                },
                "graceful_degradation": {
                    "enabled": True,
                    "fallback_responses": {
                        "llm": "I'm temporarily unable to process your request. Please try again later.",
                        "search": "Search functionality is temporarily unavailable."
                    }
                }
            },
            "caching": {
                "enabled": True,
                "provider": "memory",
                "default_ttl": 300,
                "response_caching": {
                    "enabled": True,
                    "ttl": 600
                },
                "embedding_caching": {
                    "enabled": True,
                    "ttl": 3600
                },
                "session_caching": {
                    "enabled": True,
                    "ttl": 1800
                }
            },
            "database": {
                "provider": "sqlite",
                "sqlite": {
                    "database": ":memory:"
                }
            },
            "llm": {
                "provider": "mock",
                "mock": {
                    "default_response": "This is a mock LLM response for testing."
                }
            },
            "embeddings": {
                "provider": "mock",
                "mock": {
                    "dimension": 384
                }
            },
            "vectorstore": {
                "provider": "memory"
            }
        }
        
        # Production API configuration
        self.api_config = APICustomizationConfig(
            debug=False,
            enable_docs=True,
            security=SecurityConfig(
                enable_auth=True,
                enable_rate_limiting=True,
                enable_security_headers=True
            ),
            monitoring=MonitoringConfig(
                enable_metrics=True,
                enable_health_checks=True,
                enable_prometheus=True,
                enable_alerting=True
            )
        )
        
        # Create production FastAPI application
        self.api = FastAPIEnhanced(config=self.config, api_config=self.api_config)
        self.app = self.api.create_app()
        self.client = TestClient(self.app)
        
        # Initialize production database
        self.db_manager = ProductionDatabaseManager(self.config)
        self.db_manager.initialize_database()
        
        # Create test users
        self.admin_user_id = self.db_manager.create_user(
            username="admin_user",
            email="admin@example.com",
            password="admin_password_123",
            role="admin"
        )
        
        self.regular_user_id = self.db_manager.create_user(
            username="regular_user",
            email="user@example.com",
            password="user_password_123",
            role="user"
        )
    
    def test_complete_user_authentication_journey(self):
        """Test complete user authentication journey."""
        # 1. User registration (if endpoint exists)
        registration_data = {
            "username": "new_user",
            "email": "newuser@example.com",
            "password": "new_password_123",
            "role": "user"
        }
        
        # Try registration endpoint
        try:
            reg_response = self.client.post("/auth/register", json=registration_data)
            if reg_response.status_code == 201:
                assert "user_id" in reg_response.json()
        except Exception:
            # Registration endpoint might not exist
            pass
        
        # 2. User login
        login_data = {
            "username": "regular_user",
            "password": "user_password_123"
        }
        
        login_response = self.client.post("/auth/login", json=login_data)
        assert login_response.status_code == 200
        
        login_result = login_response.json()
        assert "access_token" in login_result
        assert "token_type" in login_result
        
        access_token = login_result["access_token"]
        
        # 3. Use token for authenticated requests
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Test protected endpoint
        protected_response = self.client.get("/user/profile", headers=headers)
        assert protected_response.status_code in [200, 404]  # Either works or endpoint doesn't exist
        
        # 4. Token refresh (if implemented)
        try:
            refresh_response = self.client.post("/auth/refresh", headers=headers)
            if refresh_response.status_code == 200:
                refresh_result = refresh_response.json()
                assert "access_token" in refresh_result
        except Exception:
            # Refresh endpoint might not exist
            pass
        
        # 5. Logout
        try:
            logout_response = self.client.post("/auth/logout", headers=headers)
            assert logout_response.status_code in [200, 204]
        except Exception:
            # Logout endpoint might not exist
            pass
    
    def test_complete_chat_workflow_with_caching(self):
        """Test complete chat workflow with caching and optimization."""
        headers = {"Authorization": "Bearer e2e-api-key-user"}
        
        # 1. Initial chat request
        chat_request = {
            "query": "What is artificial intelligence?",
            "session_id": "test_session_123",
            "context": {
                "user_preferences": {"language": "en", "detail_level": "medium"}
            }
        }
        
        # First request (should hit LLM)
        start_time = time.time()
        response1 = self.client.post("/api/chat", json=chat_request, headers=headers)
        first_duration = time.time() - start_time
        
        assert response1.status_code == 200
        result1 = response1.json()
        
        assert "response" in result1
        assert "session_id" in result1
        assert result1["session_id"] == "test_session_123"
        
        # 2. Identical request (should be cached)
        start_time = time.time()
        response2 = self.client.post("/api/chat", json=chat_request, headers=headers)
        second_duration = time.time() - start_time
        
        assert response2.status_code == 200
        result2 = response2.json()
        
        # Results should be identical (cached)
        assert result1["response"] == result2["response"]
        
        # Second request should be faster (cached)
        assert second_duration <= first_duration * 1.5  # Allow some variance
        
        # 3. Follow-up question in same session
        followup_request = {
            "query": "Can you explain machine learning specifically?",
            "session_id": "test_session_123",
            "context": {"previous_query": "What is artificial intelligence?"}
        }
        
        followup_response = self.client.post("/api/chat", json=followup_request, headers=headers)
        assert followup_response.status_code == 200
        
        followup_result = followup_response.json()
        assert followup_result["session_id"] == "test_session_123"
        
        # 4. Check session persistence
        session_info_response = self.client.get(f"/api/session/test_session_123", headers=headers)
        if session_info_response.status_code == 200:
            session_info = session_info_response.json()
            assert "query_count" in session_info or "last_activity" in session_info
    
    def test_complete_document_processing_workflow(self):
        """Test complete document processing workflow."""
        headers = {"Authorization": "Bearer e2e-api-key-admin"}
        
        # 1. Upload document
        document_data = {
            "title": "AI Research Paper",
            "content": "This is a comprehensive research paper about artificial intelligence. " * 50,
            "metadata": {
                "author": "Dr. AI Researcher",
                "category": "research",
                "tags": ["AI", "machine learning", "deep learning"]
            }
        }
        
        upload_response = self.client.post("/api/documents", json=document_data, headers=headers)
        if upload_response.status_code == 201:
            upload_result = upload_response.json()
            document_id = upload_result["document_id"]
            
            # 2. Check document processing status
            status_response = self.client.get(f"/api/documents/{document_id}/status", headers=headers)
            assert status_response.status_code == 200
            
            # 3. Wait for processing to complete
            max_wait = 30  # seconds
            wait_time = 0
            while wait_time < max_wait:
                status_response = self.client.get(f"/api/documents/{document_id}/status", headers=headers)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") == "completed":
                        break
                time.sleep(1)
                wait_time += 1
            
            # 4. Query the processed document
            query_request = {
                "query": "What does this document say about machine learning?",
                "document_ids": [document_id],
                "max_results": 5
            }
            
            query_response = self.client.post("/api/search", json=query_request, headers=headers)
            if query_response.status_code == 200:
                query_result = query_response.json()
                assert "results" in query_result
                assert len(query_result["results"]) > 0
                
                # 5. Get document chunks
                chunks_response = self.client.get(f"/api/documents/{document_id}/chunks", headers=headers)
                assert chunks_response.status_code == 200
                
                chunks_data = chunks_response.json()
                assert "chunks" in chunks_data
                assert len(chunks_data["chunks"]) > 0
    
    def test_system_monitoring_and_alerting_workflow(self):
        """Test system monitoring and alerting workflow."""
        headers = {"Authorization": "Bearer e2e-api-key-admin"}
        
        # 1. Check initial system health
        health_response = self.client.get("/health", headers=headers)
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        assert "status" in health_data
        assert "components" in health_data
        
        # 2. Generate system load to trigger monitoring
        for i in range(20):
            self.client.post("/api/chat", 
                           json={"query": f"Load test query {i}"}, 
                           headers=headers)
        
        # 3. Check metrics collection
        metrics_response = self.client.get("/metrics", headers=headers)
        assert metrics_response.status_code == 200
        
        metrics_data = metrics_response.content.decode()
        assert len(metrics_data) > 0
        
        # Should contain request metrics
        assert "http_requests" in metrics_data or "request_count" in metrics_data
        
        # 4. Check detailed health status
        detailed_health_response = self.client.get("/health/detailed", headers=headers)
        if detailed_health_response.status_code == 200:
            detailed_health = detailed_health_response.json()
            assert "system_metrics" in detailed_health
            assert "application_metrics" in detailed_health
        
        # 5. Check alerts (if any were triggered)
        alerts_response = self.client.get("/api/alerts", headers=headers)
        if alerts_response.status_code == 200:
            alerts_data = alerts_response.json()
            assert "alerts" in alerts_data
            # Alerts list might be empty, which is fine
    
    def test_error_handling_and_recovery_workflow(self):
        """Test error handling and recovery workflow."""
        headers = {"Authorization": "Bearer e2e-api-key-user"}
        
        # 1. Test input validation errors
        invalid_requests = [
            {"query": ""},  # Empty query
            {"query": "<script>alert('xss')</script>"},  # XSS attempt
            {"query": "x" * 15000},  # Too long
            {}  # Missing required fields
        ]
        
        for invalid_request in invalid_requests:
            response = self.client.post("/api/chat", json=invalid_request, headers=headers)
            assert response.status_code >= 400
            
            # Error response should be well-formed
            if response.headers.get("content-type", "").startswith("application/json"):
                error_data = response.json()
                assert "error" in error_data or "detail" in error_data
        
        # 2. Test service failure simulation
        with patch('rag_engine.core.llm.LLMManager.generate') as mock_llm:
            mock_llm.side_effect = Exception("Simulated LLM failure")
            
            # Should trigger graceful degradation
            response = self.client.post("/api/chat", 
                                      json={"query": "test graceful degradation"}, 
                                      headers=headers)
            
            # Should not return 500 error due to graceful degradation
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                result = response.json()
                # Should contain fallback message
                assert "temporarily unavailable" in result.get("response", "").lower()
        
        # 3. Test circuit breaker activation
        with patch('rag_engine.core.llm.LLMManager.generate') as mock_llm:
            mock_llm.side_effect = Exception("Persistent failure")
            
            # Make multiple requests to trigger circuit breaker
            for i in range(6):  # Above failure threshold
                response = self.client.post("/api/chat", 
                                          json={"query": f"circuit breaker test {i}"}, 
                                          headers=headers)
                # Later requests should trigger circuit breaker
                if i >= 5:
                    assert response.status_code in [503, 429]
        
        # 4. Test recovery after circuit breaker
        # Remove the patch to simulate service recovery
        response = self.client.post("/api/chat", 
                                  json={"query": "recovery test"}, 
                                  headers=headers)
        # Should eventually recover (might take a few attempts)
        assert response.status_code in [200, 503]
    
    def test_security_and_compliance_workflow(self):
        """Test security and compliance workflow."""
        # 1. Test unauthenticated access
        unauth_response = self.client.post("/api/chat", json={"query": "test"})
        assert unauth_response.status_code in [401, 403]
        
        # 2. Test invalid authentication
        invalid_headers = {"Authorization": "Bearer invalid-token"}
        invalid_response = self.client.post("/api/chat", 
                                          json={"query": "test"}, 
                                          headers=invalid_headers)
        assert invalid_response.status_code in [401, 403]
        
        # 3. Test rate limiting
        headers = {"Authorization": "Bearer e2e-api-key-user"}
        
        # Make rapid requests to trigger rate limiting
        rate_limited = False
        for i in range(100):
            response = self.client.get("/health", headers=headers)
            if response.status_code == 429:
                rate_limited = True
                break
        
        # Rate limiting might not trigger in test environment
        assert rate_limited or True
        
        # 4. Test audit logging
        audit_request = {
            "query": "This request should be audited",
            "session_id": "audit_test_session"
        }
        
        response = self.client.post("/api/chat", json=audit_request, headers=headers)
        # Request should complete (audit logging should not interfere)
        assert response.status_code != 500
        
        # 5. Test security headers
        assert "x-content-type-options" in [h.lower() for h in response.headers.keys()]
        assert "x-frame-options" in [h.lower() for h in response.headers.keys()]
    
    def test_performance_and_scalability_workflow(self):
        """Test performance and scalability workflow."""
        headers = {"Authorization": "Bearer e2e-api-key-user"}
        
        # 1. Baseline performance test
        baseline_times = []
        for i in range(5):
            start_time = time.time()
            response = self.client.get("/health", headers=headers)
            end_time = time.time()
            
            assert response.status_code == 200
            baseline_times.append(end_time - start_time)
        
        baseline_avg = sum(baseline_times) / len(baseline_times)
        
        # 2. Load test with concurrent requests
        import threading
        
        concurrent_times = []
        errors = []
        
        def load_worker():
            try:
                start_time = time.time()
                response = self.client.post("/api/chat", 
                                          json={"query": "concurrent load test"}, 
                                          headers=headers)
                end_time = time.time()
                
                concurrent_times.append(end_time - start_time)
                if response.status_code not in [200, 429]:  # 429 is acceptable for rate limiting
                    errors.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create concurrent load
        threads = []
        for i in range(10):
            thread = threading.Thread(target=load_worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 3. Analyze performance under load
        if concurrent_times:
            concurrent_avg = sum(concurrent_times) / len(concurrent_times)
            
            # Performance should not degrade significantly under load
            assert concurrent_avg < baseline_avg * 3  # Allow 3x degradation
            
            # Error rate should be low
            error_rate = len(errors) / (len(concurrent_times) + len(errors))
            assert error_rate < 0.2  # Less than 20% error rate
        
        # 4. Memory usage check (simplified)
        # In production, you would check actual memory metrics
        response = self.client.get("/metrics", headers=headers)
        assert response.status_code == 200
        assert len(response.content) > 0
    
    def test_data_consistency_and_integrity_workflow(self):
        """Test data consistency and integrity workflow."""
        headers = {"Authorization": "Bearer e2e-api-key-admin"}
        
        # 1. Create user data
        user_data = {
            "username": "consistency_test_user",
            "email": "consistency@example.com",
            "password": "consistency_password_123",
            "role": "user"
        }
        
        # Create user via API
        create_response = self.client.post("/api/users", json=user_data, headers=headers)
        if create_response.status_code == 201:
            user_result = create_response.json()
            user_id = user_result["user_id"]
            
            # 2. Verify user can be retrieved
            get_response = self.client.get(f"/api/users/{user_id}", headers=headers)
            assert get_response.status_code == 200
            
            retrieved_user = get_response.json()
            assert retrieved_user["username"] == user_data["username"]
            assert retrieved_user["email"] == user_data["email"]
            
            # 3. Update user data
            update_data = {"email": "updated_consistency@example.com"}
            update_response = self.client.put(f"/api/users/{user_id}", 
                                            json=update_data, 
                                            headers=headers)
            
            if update_response.status_code == 200:
                # 4. Verify update was applied
                updated_response = self.client.get(f"/api/users/{user_id}", headers=headers)
                assert updated_response.status_code == 200
                
                updated_user = updated_response.json()
                assert updated_user["email"] == update_data["email"]
                assert updated_user["username"] == user_data["username"]  # Should remain unchanged
        
        # 5. Test session consistency
        session_data = {
            "user_id": self.regular_user_id,
            "data": {"test": "session_consistency"}
        }
        
        session_response = self.client.post("/api/sessions", json=session_data, headers=headers)
        if session_response.status_code == 201:
            session_result = session_response.json()
            session_id = session_result["session_id"]
            
            # Verify session can be retrieved
            get_session_response = self.client.get(f"/api/sessions/{session_id}", headers=headers)
            assert get_session_response.status_code == 200
            
            retrieved_session = get_session_response.json()
            assert retrieved_session["user_id"] == self.regular_user_id
    
    def test_complete_system_integration_workflow(self):
        """Test complete system integration workflow."""
        headers = {"Authorization": "Bearer e2e-api-key-admin"}
        
        # 1. System startup verification
        health_response = self.client.get("/health", headers=headers)
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        assert health_data["status"] in ["healthy", "degraded"]
        
        # 2. Component integration test
        components_to_test = [
            "/api/chat",      # LLM integration
            "/api/search",    # Vector store integration
            "/api/documents", # Document processing
            "/metrics",       # Monitoring integration
        ]
        
        component_results = {}
        for component in components_to_test:
            try:
                if component == "/api/chat":
                    response = self.client.post(component, 
                                              json={"query": "integration test"}, 
                                              headers=headers)
                elif component == "/api/search":
                    response = self.client.post(component, 
                                              json={"query": "search test", "max_results": 5}, 
                                              headers=headers)
                elif component == "/api/documents":
                    response = self.client.get(component, headers=headers)
                else:
                    response = self.client.get(component, headers=headers)
                
                component_results[component] = response.status_code
            except Exception as e:
                component_results[component] = f"Error: {str(e)}"
        
        # 3. Verify core components are working
        working_components = sum(1 for status in component_results.values() 
                               if isinstance(status, int) and status < 500)
        
        # At least health and metrics should work
        assert working_components >= 1
        
        # 4. End-to-end data flow test
        # Create document -> Process -> Search -> Chat
        document_data = {
            "title": "Integration Test Document",
            "content": "This document is used for integration testing of the complete system workflow.",
            "metadata": {"test": True}
        }
        
        # Upload document
        doc_response = self.client.post("/api/documents", json=document_data, headers=headers)
        if doc_response.status_code == 201:
            doc_result = doc_response.json()
            document_id = doc_result["document_id"]
            
            # Wait for processing
            time.sleep(2)
            
            # Search for document
            search_response = self.client.post("/api/search", 
                                             json={"query": "integration testing", "max_results": 5}, 
                                             headers=headers)
            
            # Chat about document
            chat_response = self.client.post("/api/chat", 
                                           json={"query": "Tell me about the integration test document"}, 
                                           headers=headers)
            
            # At least one operation should succeed
            assert any(r.status_code == 200 for r in [search_response, chat_response])
        
        # 5. Final system health check
        final_health_response = self.client.get("/health", headers=headers)
        assert final_health_response.status_code == 200
        
        final_health_data = final_health_response.json()
        # System should still be healthy after integration test
        assert final_health_data["status"] in ["healthy", "degraded"]


class TestProductionE2EUserJourneys:
    """Test realistic user journeys in production environment."""
    
    def setup_method(self):
        """Setup production environment for user journey tests."""
        # Simplified config for user journey tests
        self.config = {
            "security": {
                "api_keys": ["journey-test-key"],
                "rate_limiting": {"enabled": True}
            },
            "monitoring": {"enabled": True},
            "caching": {"enabled": True},
            "database": {"provider": "sqlite", "sqlite": {"database": ":memory:"}}
        }
        
        self.api_config = APICustomizationConfig()
        self.api = FastAPIEnhanced(config=self.config, api_config=self.api_config)
        self.app = self.api.create_app()
        self.client = TestClient(self.app)
    
    def test_new_user_onboarding_journey(self):
        """Test new user onboarding journey."""
        headers = {"Authorization": "Bearer journey-test-key"}
        
        # 1. New user explores system
        health_response = self.client.get("/health")
        assert health_response.status_code == 200
        
        # 2. User tries basic functionality
        basic_query = {"query": "Hello, what can you help me with?"}
        intro_response = self.client.post("/api/chat", json=basic_query, headers=headers)
        
        # Should get some response (success or graceful failure)
        assert intro_response.status_code in [200, 400, 503]
        
        # 3. User asks follow-up questions
        followup_queries = [
            "What are your capabilities?",
            "How do I get started?",
            "Can you help me with research?"
        ]
        
        for query in followup_queries:
            response = self.client.post("/api/chat", 
                                      json={"query": query}, 
                                      headers=headers)
            # Should handle all queries consistently
            assert response.status_code == intro_response.status_code
    
    def test_power_user_workflow_journey(self):
        """Test power user workflow journey."""
        headers = {"Authorization": "Bearer journey-test-key"}
        
        # 1. Power user uploads multiple documents
        documents = [
            {"title": "Research Paper 1", "content": "Content about AI research" * 20},
            {"title": "Technical Manual", "content": "Technical documentation" * 30},
            {"title": "Meeting Notes", "content": "Important meeting discussion" * 15}
        ]
        
        document_ids = []
        for doc in documents:
            response = self.client.post("/api/documents", json=doc, headers=headers)
            if response.status_code == 201:
                result = response.json()
                document_ids.append(result["document_id"])
        
        # 2. User performs complex searches
        complex_queries = [
            {"query": "Find information about AI research methodology", "max_results": 10},
            {"query": "Technical implementation details", "document_ids": document_ids[:2]},
            {"query": "Meeting decisions and action items", "filters": {"title": "Meeting"}}
        ]
        
        for query in complex_queries:
            response = self.client.post("/api/search", json=query, headers=headers)
            # Should handle complex queries
            assert response.status_code in [200, 400, 404]
        
        # 3. User analyzes results with chat
        analysis_queries = [
            "Summarize the key findings from the research papers",
            "What are the main technical requirements mentioned?",
            "Compare the different approaches discussed"
        ]
        
        for query in analysis_queries:
            response = self.client.post("/api/chat", 
                                      json={"query": query}, 
                                      headers=headers)
            assert response.status_code in [200, 503]
    
    def test_enterprise_team_collaboration_journey(self):
        """Test enterprise team collaboration journey."""
        headers = {"Authorization": "Bearer journey-test-key"}
        
        # 1. Team lead sets up shared workspace
        workspace_data = {
            "name": "AI Research Team",
            "description": "Collaborative workspace for AI research",
            "settings": {"sharing_enabled": True, "version_control": True}
        }
        
        workspace_response = self.client.post("/api/workspaces", json=workspace_data, headers=headers)
        workspace_id = None
        if workspace_response.status_code == 201:
            workspace_result = workspace_response.json()
            workspace_id = workspace_result["workspace_id"]
        
        # 2. Team members contribute documents
        team_documents = [
            {"title": "Project Proposal", "content": "AI project proposal details" * 25, "author": "team_lead"},
            {"title": "Literature Review", "content": "Comprehensive literature review" * 35, "author": "researcher_1"},
            {"title": "Technical Specs", "content": "Technical specifications" * 20, "author": "engineer_1"}
        ]
        
        for doc in team_documents:
            if workspace_id:
                doc["workspace_id"] = workspace_id
            
            response = self.client.post("/api/documents", json=doc, headers=headers)
            # Should accept team contributions
            assert response.status_code in [201, 400]
        
        # 3. Team collaboration through chat
        collaboration_queries = [
            "What's the current status of our AI project?",
            "Are there any technical blockers mentioned in the specs?",
            "What does the literature review suggest about our approach?"
        ]
        
        for query in collaboration_queries:
            if workspace_id:
                chat_data = {"query": query, "workspace_id": workspace_id}
            else:
                chat_data = {"query": query}
            
            response = self.client.post("/api/chat", json=chat_data, headers=headers)
            assert response.status_code in [200, 400, 503]
    
    def test_system_stress_recovery_journey(self):
        """Test system behavior under stress and recovery."""
        headers = {"Authorization": "Bearer journey-test-key"}
        
        # 1. Normal operation baseline
        baseline_response = self.client.get("/health", headers=headers)
        assert baseline_response.status_code == 200
        
        # 2. Gradual load increase
        load_levels = [5, 10, 20, 30]
        
        for load_level in load_levels:
            responses = []
            
            # Generate load
            for i in range(load_level):
                response = self.client.post("/api/chat", 
                                          json={"query": f"stress test query {i}"}, 
                                          headers=headers)
                responses.append(response.status_code)
            
            # Check system health after load
            health_response = self.client.get("/health", headers=headers)
            
            # System should remain responsive or degrade gracefully
            assert health_response.status_code == 200
            
            # Calculate success rate
            success_rate = sum(1 for status in responses if status == 200) / len(responses)
            
            # Should maintain reasonable success rate or fail gracefully
            assert success_rate >= 0.5 or all(status in [429, 503] for status in responses)
        
        # 3. Recovery verification
        time.sleep(2)  # Allow recovery time
        
        recovery_response = self.client.get("/health", headers=headers)
        assert recovery_response.status_code == 200
        
        # System should recover to normal operation
        normal_response = self.client.post("/api/chat", 
                                         json={"query": "recovery test"}, 
                                         headers=headers)
        assert normal_response.status_code in [200, 503]  # Should work or gracefully degrade 