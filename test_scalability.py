#!/usr/bin/env python3
"""
Test script to verify production scalability of RAG Engine APIs.
Tests concurrent requests to ensure proper multi-worker handling.
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Tuple
import sys
import argparse


async def make_request(session: aiohttp.ClientSession, url: str, request_id: int) -> Tuple[int, float, str]:
    """Make a single request and return response time."""
    start_time = time.time()
    try:
        async with session.get(url) as response:
            await response.text()
            elapsed = time.time() - start_time
            return request_id, elapsed, f"HTTP {response.status}"
    except Exception as e:
        elapsed = time.time() - start_time
        return request_id, elapsed, f"Error: {str(e)}"


async def load_test(base_url: str, num_requests: int, concurrency: int) -> None:
    """Run concurrent load test against the API."""
    print(f"🧪 Load Testing RAG Engine API")
    print(f"📍 URL: {base_url}")
    print(f"🔢 Total Requests: {num_requests}")
    print(f"⚡ Concurrency: {concurrency}")
    print(f"{'='*50}")
    
    # Test endpoints
    endpoints = [
        "/health",
        "/status", 
        "/config",
        "/documents",
        "/chunks"
    ]
    
    results = []
    
    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for endpoint in endpoints:
            print(f"\n🎯 Testing endpoint: {endpoint}")
            url = f"{base_url}{endpoint}"
            
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(concurrency)
            
            async def bounded_request(request_id: int) -> Tuple[int, float, str]:
                async with semaphore:
                    return await make_request(session, url, request_id)
            
            # Run requests
            start_time = time.time()
            tasks = [bounded_request(i) for i in range(num_requests)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Process results
            successful = []
            failed = []
            
            for response in responses:
                if isinstance(response, Exception):
                    failed.append(str(response))
                else:
                    req_id, elapsed, status = response
                    if "HTTP 2" in status:  # 2xx status codes
                        successful.append(elapsed)
                    else:
                        failed.append(status)
            
            # Calculate metrics
            success_rate = len(successful) / num_requests * 100
            avg_response_time = statistics.mean(successful) if successful else 0
            rps = len(successful) / total_time if total_time > 0 else 0
            
            print(f"   ✅ Success Rate: {success_rate:.1f}% ({len(successful)}/{num_requests})")
            print(f"   ⏱️  Avg Response Time: {avg_response_time:.3f}s")
            print(f"   🚀 Requests/sec: {rps:.1f}")
            
            if failed:
                print(f"   ❌ Failures: {len(failed)}")
                for failure in failed[:3]:  # Show first 3 failures
                    print(f"      • {failure}")
            
            results.append({
                'endpoint': endpoint,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'rps': rps,
                'successful': len(successful),
                'failed': len(failed)
            })
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 LOAD TEST SUMMARY")
    print(f"{'='*50}")
    
    overall_success = sum(r['successful'] for r in results)
    overall_failed = sum(r['failed'] for r in results)
    overall_total = overall_success + overall_failed
    overall_success_rate = overall_success / overall_total * 100 if overall_total > 0 else 0
    
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    print(f"Total Successful: {overall_success}")
    print(f"Total Failed: {overall_failed}")
    
    avg_rps = statistics.mean([r['rps'] for r in results if r['rps'] > 0])
    print(f"Average RPS: {avg_rps:.1f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if overall_success_rate >= 95:
        print("✅ Excellent! API is handling load well.")
        if avg_rps < 10:
            print("💡 Consider increasing worker count for higher throughput.")
    elif overall_success_rate >= 80:
        print("⚠️  Good performance, but consider optimization.")
        print("💡 Monitor error logs and consider scaling workers.")
    else:
        print("❌ Poor performance detected!")
        print("💡 Increase workers, check resource limits, review error logs.")
    
    return results


async def test_chat_endpoint(base_url: str, num_requests: int = 10) -> None:
    """Test the chat endpoint with actual queries."""
    print(f"\n🤖 Testing Chat Endpoint")
    print(f"{'='*30}")
    
    test_queries = [
        "What is this system about?",
        "How does RAG work?",
        "What are the main features?",
        "Can you explain the architecture?",
        "What technologies are used?"
    ]
    
    url = f"{base_url}/chat"
    results = []
    
    connector = aiohttp.TCPConnector(limit=5)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for i, query in enumerate(test_queries[:min(len(test_queries), num_requests)]):
            start_time = time.time()
            try:
                payload = {"query": query, "session_id": f"test_{i}"}
                async with session.post(url, json=payload) as response:
                    result = await response.json()
                    elapsed = time.time() - start_time
                    
                    print(f"Query {i+1}: {query[:30]}...")
                    print(f"  Status: HTTP {response.status}")
                    print(f"  Time: {elapsed:.2f}s")
                    if response.status == 200:
                        response_text = result.get('response', 'No response')[:100]
                        print(f"  Response: {response_text}...")
                    print()
                    
                    results.append({
                        'query': query,
                        'status': response.status,
                        'time': elapsed,
                        'success': response.status == 200
                    })
                    
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"Query {i+1}: {query[:30]}... FAILED ({e})")
                results.append({
                    'query': query,
                    'status': 'error',
                    'time': elapsed,
                    'success': False
                })
    
    success_count = sum(1 for r in results if r['success'])
    avg_time = statistics.mean([r['time'] for r in results])
    
    print(f"Chat Test Results:")
    print(f"  Success Rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"  Average Response Time: {avg_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Load test RAG Engine API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--requests", type=int, default=50, help="Number of requests per endpoint")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--chat-test", action="store_true", help="Include chat endpoint testing")
    
    args = parser.parse_args()
    
    async def run_tests():
        # Basic load test
        await load_test(args.url, args.requests, args.concurrency)
        
        # Chat endpoint test (if enabled)
        if args.chat_test:
            await test_chat_endpoint(args.url, min(args.requests, 10))
    
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
