#!/usr/bin/env python
"""
Test script to verify the Django chat view with summary statistics
"""
import os
import sys
import django
import json

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai.settings')
django.setup()

from django.test import Client
from django.contrib.auth.models import User
from analyticabd.models import UserDataset

def test_chat_view():
    """Test the Django chat view"""
    print("🔍 Testing Django Chat View...")
    
    try:
        # Create a test client
        client = Client()
        
        # Get the first dataset and its owner
        datasets = UserDataset.objects.all()
        if not datasets.exists():
            print("❌ No datasets found in database")
            return False
        
        dataset = datasets.first()
        print(f"✅ Using dataset: {dataset.name}")
        print(f"   Dataset owner: {dataset.user.username}")
        
        # Login as the dataset owner
        client.force_login(dataset.user)
        
        # Test the chat endpoint with summary statistics context
        chat_data = {
            'message': 'what do you think of the data?',
            'session_id': 'test_session_123',
            'context': {
                'dataset_id': dataset.id,
                'analysis_type': 'summary_statistics'
            }
        }
        
        print(f"✅ Testing chat endpoint with dataset_id: {dataset.id}")
        
        # Make the request
        response = client.post('/api/v1/chat/send_message/', 
                             data=json.dumps(chat_data),
                             content_type='application/json')
        
        print(f"✅ Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = json.loads(response.content)
            print("✅ Chat view working correctly!")
            print(f"   Response length: {len(response_data.get('response', ''))} characters")
            print(f"   Tool executed: {response_data.get('tool_executed')}")
            
            # Print first 300 characters of response
            response_text = response_data.get('response', '')
            response_preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
            print(f"   Response preview: {response_preview}")
            
            return True
        else:
            print(f"❌ Chat view error: {response.content}")
            return False
        
    except Exception as e:
        print(f"❌ Error testing chat view: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Django Chat View Test...")
    success = test_chat_view()
    
    if success:
        print("✅ Django chat view is working correctly!")
    else:
        print("❌ Django chat view has issues!")
    
    print("\n✨ Test completed!")
