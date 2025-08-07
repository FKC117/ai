#!/usr/bin/env python
"""
Test script to verify summary statistics function
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai.settings')
django.setup()

from analyticabd.views import get_summary_statistics_data
from analyticabd.models import UserDataset

def test_summary_statistics():
    """Test the summary statistics function"""
    print("🔍 Testing Summary Statistics Function...")
    
    try:
        # Get the first dataset
        datasets = UserDataset.objects.all()
        if not datasets.exists():
            print("❌ No datasets found in database")
            return False
        
        dataset = datasets.first()
        print(f"✅ Found dataset: {dataset.name}")
        print(f"   File path: {dataset.file.path}")
        print(f"   File exists: {os.path.exists(dataset.file.path)}")
        
        # Test the summary statistics function
        summary_stats = get_summary_statistics_data(dataset.id)
        
        print("✅ Summary statistics generated successfully!")
        print(f"   Variables: {len(summary_stats['variable_summary'])}")
        print(f"   Data quality metrics: {len(summary_stats['data_quality'])}")
        
        # Print some sample data
        print("\n📊 Sample Variable Summary:")
        for var_name, stats in list(summary_stats['variable_summary'].items())[:3]:
            print(f"   {var_name}: {stats['type']} - Count: {stats['count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing summary statistics: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Summary Statistics Test...")
    success = test_summary_statistics()
    
    if success:
        print("✅ Summary statistics function is working correctly!")
    else:
        print("❌ Summary statistics function has issues!")
    
    print("\n✨ Test completed!")
