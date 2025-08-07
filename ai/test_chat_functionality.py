#!/usr/bin/env python
"""
Test script to verify AI chat functionality with summary statistics
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai.settings')
django.setup()

from analyticabd.ai.tool_executor import ToolExecutor
from analyticabd.models import UserDataset, User
from django.contrib.auth.models import User as AuthUser

def test_chat_with_summary_statistics():
    """Test the AI chat with summary statistics"""
    print("🔍 Testing AI Chat with Summary Statistics...")
    
    try:
        # Get a test user
        user, created = AuthUser.objects.get_or_create(
            username='testuser',
            defaults={'email': 'test@example.com'}
        )
        
        # Get the first dataset
        datasets = UserDataset.objects.all()
        if not datasets.exists():
            print("❌ No datasets found in database")
            return False
        
        dataset = datasets.first()
        print(f"✅ Using dataset: {dataset.name}")
        
        # Create tool executor
        tool_executor = ToolExecutor()
        
        # Test context with summary statistics
        context = {
            'dataset_id': dataset.id,
            'dataset_name': dataset.name,
            'dataset_info': {
                'name': dataset.name,
                'rows': dataset.rows,
                'columns': dataset.columns,
                'numeric_columns': dataset.numeric_columns,
                'categorical_columns': dataset.categorical_columns
            }
        }
        
        # Test message that should trigger summary statistics
        test_message = "what do you think of the data?"
        
        print(f"✅ Testing message: '{test_message}'")
        
        # Process the message
        result = tool_executor.process_chat_message(
            user_id=user.id,
            message=test_message,
            session_id='test_session_123',
            context=context
        )
        
        if result.get('error'):
            print(f"❌ Error in chat processing: {result['error']}")
            return False
        
        print("✅ Chat processing completed successfully!")
        print(f"   Response length: {len(result['response'])} characters")
        print(f"   Tool executed: {result.get('tool_executed')}")
        
        # Print first 200 characters of response
        response_preview = result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
        print(f"   Response preview: {response_preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing chat functionality: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting AI Chat Test...")
    success = test_chat_with_summary_statistics()
    
    if success:
        print("✅ AI chat functionality is working correctly!")
    else:
        print("❌ AI chat functionality has issues!")
    
    print("\n✨ Test completed!")
