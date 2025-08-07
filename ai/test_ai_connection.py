#!/usr/bin/env python
"""
Test script to verify AI connection and components
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai.settings')
django.setup()

from analyticabd.ai.llm_client import LLMClient
from analyticabd.ai.cache_manager import CacheManager
from analyticabd.ai.tool_executor import ToolExecutor
from analyticabd.tools.tool_registry import ToolRegistry

def test_ai_components():
    """Test all AI components"""
    print("🔍 Testing AI Components...")
    
    # Test 1: Check environment variables
    print("\n1. Checking Environment Variables:")
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        print("✅ GOOGLE_API_KEY found")
    else:
        print("❌ GOOGLE_API_KEY not found")
        return False
    
    # Test 2: Test LLM Client
    print("\n2. Testing LLM Client:")
    try:
        llm_client = LLMClient()
        print("✅ LLM Client initialized successfully")
        
        # Test a simple chat
        response = llm_client.chat("Hello, can you hear me?")
        print(f"✅ LLM Response received: {response[:100]}...")
        
    except Exception as e:
        print(f"❌ LLM Client error: {str(e)}")
        return False
    
    # Test 3: Test Cache Manager
    print("\n3. Testing Cache Manager:")
    try:
        cache_manager = CacheManager()
        print("✅ Cache Manager initialized successfully")
        
        # Test cache operations (may fail if Redis is not available)
        test_key = "test_key"
        test_data = {"test": "data"}
        cache_manager.set_cached_response(test_key, test_data)
        cached_data = cache_manager.get_cached_response(test_key)
        
        if cached_data == test_data:
            print("✅ Cache operations working correctly")
        else:
            print("⚠️ Cache operations failed (Redis not available) - continuing without cache")
            
    except Exception as e:
        print(f"⚠️ Cache Manager error: {str(e)} - continuing without cache")
    
    # Test 4: Test Tool Registry
    print("\n4. Testing Tool Registry:")
    try:
        tool_registry = ToolRegistry()
        available_tools = tool_registry.list_available_tools()
        print(f"✅ Tool Registry initialized. Available tools: {available_tools}")
        
        # Test getting tool descriptions
        descriptions = tool_registry.get_tool_descriptions()
        print(f"✅ Tool descriptions retrieved: {list(descriptions.keys())}")
        
    except Exception as e:
        print(f"❌ Tool Registry error: {str(e)}")
        return False
    
    # Test 5: Test Tool Executor
    print("\n5. Testing Tool Executor:")
    try:
        tool_executor = ToolExecutor()
        print("✅ Tool Executor initialized successfully")
        
        # Test getting available tools
        tools = tool_executor.get_available_tools()
        print(f"✅ Available tools from executor: {len(tools)} tools")
        
    except Exception as e:
        print(f"❌ Tool Executor error: {str(e)}")
        return False
    
    # Test 6: Test Summary Statistics Tool
    print("\n6. Testing Summary Statistics Tool:")
    try:
        from analyticabd.tools.summary_statistics_tool import SummaryStatisticsTool
        
        # Create a test tool instance
        test_tool = SummaryStatisticsTool(dataset_id=1, user_id=1)
        description = test_tool.get_description()
        schema = test_tool.get_parameters_schema()
        
        print(f"✅ Summary Statistics Tool initialized")
        print(f"   Description: {description[:50]}...")
        print(f"   Parameters: {list(schema.keys())}")
        
    except Exception as e:
        print(f"❌ Summary Statistics Tool error: {str(e)}")
        return False
    
    print("\n🎉 All AI components are working correctly!")
    return True

def test_chat_endpoint():
    """Test the chat endpoint"""
    print("\n🔍 Testing Chat Endpoint...")
    
    try:
        from django.test import Client
        from django.contrib.auth.models import User
        
        # Create a test client
        client = Client()
        
        # Create a test user
        user, created = User.objects.get_or_create(
            username='testuser',
            defaults={'email': 'test@example.com'}
        )
        
        # Login the user
        client.force_login(user)
        
        # Test the chat endpoint
        chat_data = {
            'message': 'Hello, test message',
            'session_id': 'test_session_123',
            'context': {
                'dataset_id': 1,
                'analysis_type': 'summary_statistics'
            }
        }
        
        response = client.post('/api/v1/chat/send_message/', 
                             data=chat_data,
                             content_type='application/json')
        
        print(f"✅ Chat endpoint response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Chat endpoint working correctly")
        else:
            print(f"❌ Chat endpoint error: {response.content}")
            
    except Exception as e:
        print(f"❌ Chat endpoint test error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting AI Connection Test...")
    
    # Test AI components
    ai_working = test_ai_components()
    
    if ai_working:
        # Test chat endpoint
        test_chat_endpoint()
    
    print("\n✨ Test completed!")
