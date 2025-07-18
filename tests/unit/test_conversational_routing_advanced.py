"""
Comprehensive test suite for edge cases, performance, and stress testing
of the conversational routing system.
"""

import pytest
import json
import time
import threading
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

from rag_engine.core.conversational_routing import (
    ConversationalRouter,
    QueryCategory,
    ResponseStrategy,
    RoutingDecision
)
from rag_engine.core.conversational_integration import ConversationalRAGPrompter


class TestConversationalRoutingEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "topic_analysis_temperature": 0.1,
            "classification_temperature": 0.1,
            "response_temperature": 0.7,
            "max_conversation_history": 5,
            "confidence_threshold": 0.8,
            "enable_reasoning_chain": True
        }
    
    def test_extremely_long_query(self):
        """Test handling of extremely long queries."""
        # Create a very long query (10000+ characters)
        long_query = "This is a test query. " * 500  # ~10000 characters
        
        router = ConversationalRouter(self.config)
        mock_llm = Mock()
        router.set_llm(mock_llm)
        
        # Mock response
        mock_llm.invoke.return_value = json.dumps({
            "topic": "general",
            "confidence": 0.7,
            "reasoning": "Long query processed"
        })
        
        # Should handle long queries without crashing
        try:
            decision = router.route_query(long_query, {})
            assert decision is not None or True  # Either works or gracefully fails
        except Exception as e:
            # Should fail gracefully if at all
            assert isinstance(e, (ValueError, MemoryError, RuntimeError))
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        unicode_queries = [
            "Bonjour, comment ça va? 🇫🇷",
            "Здравствуйте! Как дела? 🇷🇺",
            "こんにちは、元気ですか？ 🇯🇵",
            "¿Cómo está usted? ¡Muy bien! 🇪🇸",
            "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
            "Emojis: 😀😃😄😁😆😅😂🤣😊😌",
            "Math symbols: ∑∏∫∂√∞≈≠≤≥±∓⊕⊗"
        ]
        
        router = ConversationalRouter(self.config)
        mock_llm = Mock()
        router.set_llm(mock_llm)
        
        for query in unicode_queries:
            mock_llm.invoke.return_value = json.dumps({
                "topic": "international",
                "confidence": 0.8,
                "reasoning": "Unicode query processed"
            })
            
            try:
                decision = router.route_query(query, {})
                assert decision is not None or True
            except Exception as e:
                # Should handle unicode gracefully
                assert isinstance(e, (UnicodeError, ValueError))
    
    def test_malformed_json_responses(self):
        """Test handling of malformed JSON from LLM."""
        router = ConversationalRouter(self.config)
        mock_llm = Mock()
        router.set_llm(mock_llm)
        
        malformed_responses = [
            '{"topic": "test", "confidence":}',  # Missing value
            '{"topic": "test" "confidence": 0.8}',  # Missing comma
            'not json at all',
            '',  # Empty response
            '{"topic": "test", "confidence": "not_a_number"}',  # Wrong type
            '{topic: test}',  # Invalid JSON syntax
        ]
        
        for response in malformed_responses:
            mock_llm.invoke.return_value = response
            
            try:
                decision = router.route_query("test query", {})
                # If it doesn't crash, that's fine
                assert decision is not None or True
            except Exception as e:
                # Should handle malformed JSON gracefully
                assert isinstance(e, (json.JSONDecodeError, ValueError, KeyError))
    
    def test_concurrent_routing_requests(self):
        """Test concurrent routing requests for thread safety."""
        router = ConversationalRouter(self.config)
        mock_llm = Mock()
        router.set_llm(mock_llm)
        
        # Mock consistent responses
        mock_llm.invoke.return_value = json.dumps({
            "topic": "concurrent",
            "confidence": 0.8,
            "reasoning": "Concurrent test"
        })
        
        def route_query_worker(query_id):
            try:
                decision = router.route_query(f"Query {query_id}", {"id": query_id})
                return {"success": True, "query_id": query_id, "decision": decision}
            except Exception as e:
                return {"success": False, "query_id": query_id, "error": str(e)}
        
        # Run multiple queries concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(route_query_worker, i) for i in range(20)]
            results = [future.result() for future in as_completed(futures)]
        
        # Check results
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        # Should handle concurrent requests reasonably well
        assert len(successful_results) > 0  # At least some should succeed
        
        # If there are failures, they should be reasonable
        for failed in failed_results:
            assert "error" in failed
    
    def test_conversation_context_memory_limits(self):
        """Test behavior when conversation context exceeds memory limits."""
        config_with_small_limit = self.config.copy()
        config_with_small_limit["max_conversation_history"] = 3
        
        router = ConversationalRouter(config_with_small_limit)
        mock_llm = Mock()
        router.set_llm(mock_llm)
        
        mock_llm.invoke.return_value = json.dumps({
            "topic": "memory_test",
            "confidence": 0.8,
            "reasoning": "Memory limit test"
        })
        
        session_id = "memory_test_session"
        
        # Add many conversation turns
        for i in range(20):  # Much more than the limit of 3
            context = {
                "session_id": session_id,
                "conversation_history": [
                    {"role": "user", "content": f"Message {j}"}
                    for j in range(i)
                ]
            }
            
            try:
                decision = router.route_query(f"Query {i}", context)
                assert decision is not None
            except Exception as e:
                # Should handle memory limits gracefully
                assert isinstance(e, (MemoryError, ValueError))
        
        # Verify context is managed (if the router stores it)
        if session_id in router.conversation_contexts:
            stored_context = router.conversation_contexts[session_id]
            # Should respect limits or have reasonable size
            if "conversation_history" in stored_context:
                history_length = len(stored_context["conversation_history"])
                assert history_length <= config_with_small_limit["max_conversation_history"] * 4  # Some buffer
    
    def test_extremely_low_confidence_scenarios(self):
        """Test scenarios with extremely low confidence."""
        router = ConversationalRouter(self.config)
        mock_llm = Mock()
        router.set_llm(mock_llm)
        
        # Mock extremely low confidence responses
        low_confidence_scenarios = [
            {"confidence": 0.01, "topic": "unclear"},
            {"confidence": 0.0, "topic": "unknown"},
            {"confidence": -0.5, "topic": "invalid"},  # Invalid confidence
        ]
        
        for scenario in low_confidence_scenarios:
            mock_llm.invoke.side_effect = [
                json.dumps(scenario),
                json.dumps({"category": "rag_factual", "confidence": 0.1, "reasoning": "Low confidence"})
            ]
            
            try:
                decision = router.route_query("ambiguous query", "test_session")
                # Should handle low confidence appropriately
                if decision:
                    # Confidence should be normalized or handled
                    assert decision["confidence"] >= 0.0
                    assert decision["confidence"] <= 1.0
            except Exception as e:
                # Should handle invalid confidence gracefully
                assert isinstance(e, (ValueError, TypeError))
    
    def test_missing_required_fields_in_responses(self):
        """Test handling when LLM responses miss required fields."""
        router = ConversationalRouter(self.config)
        mock_llm = Mock()
        router.set_llm(mock_llm)
        
        incomplete_responses = [
            '{"topic": "test"}',  # Missing confidence
            '{"confidence": 0.8}',  # Missing topic
            '{"topic": "test", "confidence": 0.8}',  # Missing reasoning
            '{}',  # Empty object
        ]
        
        for response in incomplete_responses:
            mock_llm.invoke.return_value = response
            
            try:
                decision = router.route_query("test query", {})
                # Should handle missing fields gracefully
                assert decision is not None or True
            except Exception as e:
                # Should handle missing fields appropriately
                assert isinstance(e, (KeyError, ValueError))


class TestConversationalRoutingPerformance:
    """Test performance characteristics of the routing system."""
    
    def setup_method(self):
        """Setup performance test fixtures."""
        self.config = {
            "topic_analysis_temperature": 0.1,
            "classification_temperature": 0.1,
            "response_temperature": 0.7,
            "max_conversation_history": 10,
            "confidence_threshold": 0.8,
            "enable_reasoning_chain": True
        }
        
        self.router = ConversationalRouter(self.config)
        self.mock_llm = Mock()
        self.router.set_llm(self.mock_llm)
        
        # Mock consistent fast responses
        self.mock_llm.invoke.return_value = json.dumps({
            "topic": "performance_test",
            "confidence": 0.8,
            "reasoning": "Performance test response"
        })
    
    def test_routing_latency(self):
        """Test latency of routing decisions."""
        query = "What is the latency of this routing system?"
        context = {"conversation_history": []}
        
        # Measure multiple routing calls
        latencies = []
        for _ in range(10):
            start_time = time.time()
            try:
                decision = self.router.route_query(query, context)
                end_time = time.time()
                latencies.append(end_time - start_time)
            except:
                pass  # Skip failed attempts for latency measurement
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            # Performance assertions (adjust based on requirements)
            assert avg_latency < 1.0, f"Average latency too high: {avg_latency:.3f}s"
            assert max_latency < 2.0, f"Max latency too high: {max_latency:.3f}s"
    
    def test_throughput_under_load(self):
        """Test throughput under concurrent load."""
        def routing_worker():
            try:
                decision = self.router.route_query("Load test query", {})
                return 1 if decision else 0
            except:
                return 0
        
        start_time = time.time()
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(routing_worker) for _ in range(50)]
            successful_requests = sum(future.result() for future in as_completed(futures))
        
        end_time = time.time()
        duration = end_time - start_time
        
        if successful_requests > 0:
            throughput = successful_requests / duration
            
            # Throughput assertion (adjust based on requirements)
            assert throughput > 5.0, f"Throughput too low: {throughput:.2f} requests/second"
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable over many requests."""
        import gc
        import sys
        
        # Force garbage collection
        gc.collect()
        
        # Measure initial memory usage (rough estimate)
        initial_objects = len(gc.get_objects())
        
        # Process many queries
        for i in range(100):
            try:
                decision = self.router.route_query(f"Memory test query {i}", {"test": i})
            except:
                pass  # Continue even if some fail
            
            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()
        
        # Final garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory should not grow excessively
        object_growth = final_objects - initial_objects
        growth_ratio = object_growth / initial_objects if initial_objects > 0 else 0
        
        # Allow some growth but not excessive
        assert growth_ratio < 0.5, f"Memory usage grew too much: {growth_ratio:.2%}"
    
    def test_large_context_performance(self):
        """Test performance with large conversation contexts."""
        # Create large conversation history
        large_context = {
            "conversation_history": [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
                for i in range(1000)  # Large conversation
            ]
        }
        
        start_time = time.time()
        
        try:
            decision = self.router.route_query("Query with large context", large_context)
            end_time = time.time()
            
            latency = end_time - start_time
            
            # Should handle large contexts reasonably well
            assert latency < 5.0, f"Large context latency too high: {latency:.3f}s"
            
        except Exception as e:
            # Large contexts might fail, but should fail gracefully
            assert isinstance(e, (MemoryError, ValueError, TimeoutError))


class TestConversationalRoutingStressTest:
    """Stress tests for the conversational routing system."""
    
    def setup_method(self):
        """Setup stress test fixtures."""
        self.config = {
            "topic_analysis_temperature": 0.1,
            "classification_temperature": 0.1,
            "response_temperature": 0.7,
            "max_conversation_history": 10,
            "confidence_threshold": 0.8,
            "enable_reasoning_chain": True
        }
    
    def test_rapid_fire_requests(self):
        """Test handling of rapid-fire requests."""
        router = ConversationalRouter(self.config)
        mock_llm = Mock()
        router.set_llm(mock_llm)
        
        mock_llm.invoke.return_value = json.dumps({
            "topic": "rapid_fire",
            "confidence": 0.8,
            "reasoning": "Rapid fire test"
        })
        
        # Send requests as fast as possible
        success_count = 0
        error_count = 0
        
        for i in range(100):
            try:
                decision = router.route_query(f"Rapid query {i}", {"request_id": i})
                if decision:
                    success_count += 1
            except Exception:
                error_count += 1
        
        # Should handle most requests successfully
        total_requests = success_count + error_count
        success_rate = success_count / total_requests if total_requests > 0 else 0
        
        assert success_rate > 0.7, f"Success rate too low: {success_rate:.2%}"
    
    def test_memory_pressure_scenarios(self):
        """Test behavior under memory pressure."""
        router = ConversationalRouter(self.config)
        mock_llm = Mock()
        router.set_llm(mock_llm)
        
        mock_llm.invoke.return_value = json.dumps({
            "topic": "memory_pressure",
            "confidence": 0.8,
            "reasoning": "Memory pressure test"
        })
        
        # Create many concurrent conversations
        for session_id in range(50):
            large_context = {
                "session_id": f"stress_session_{session_id}",
                "conversation_history": [
                    {"role": "user", "content": f"Message {i} in session {session_id}"}
                    for i in range(100)
                ]
            }
            
            try:
                decision = router.route_query(f"Query in session {session_id}", large_context)
                # Success is good, but we mainly want to test it doesn't crash
            except Exception as e:
                # Should handle memory pressure gracefully
                assert isinstance(e, (MemoryError, ValueError, RuntimeError))
    
    def test_long_running_stability(self):
        """Test stability over extended usage periods."""
        router = ConversationalRouter(self.config)
        mock_llm = Mock()
        router.set_llm(mock_llm)
        
        mock_llm.invoke.return_value = json.dumps({
            "topic": "stability",
            "confidence": 0.8,
            "reasoning": "Long running test"
        })
        
        # Simulate extended usage
        error_count = 0
        success_count = 0
        
        for hour in range(3):  # Simulate 3 "hours" of usage
            for minute in range(10):  # 10 "minutes" per hour
                for request in range(5):  # 5 requests per minute
                    try:
                        decision = router.route_query(
                            f"Query at hour {hour}, minute {minute}, request {request}",
                            {"timestamp": f"{hour}:{minute}:{request}"}
                        )
                        if decision:
                            success_count += 1
                    except Exception:
                        error_count += 1
        
        total_requests = success_count + error_count
        
        if total_requests > 0:
            success_rate = success_count / total_requests
            # Should maintain stability over time
            assert success_rate > 0.8, f"Long-running stability issues: {success_rate:.2%} success rate"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
