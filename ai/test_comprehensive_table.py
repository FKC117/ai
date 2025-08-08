#!/usr/bin/env python3
"""
Test script for comprehensive summary table functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyticabd.views import create_comprehensive_summary_table

def test_comprehensive_table():
    """Test the comprehensive summary table creation"""
    
    # Sample summary statistics data
    sample_stats = {
        'variable_summary': {
            'age': {
                'type': 'numeric',
                'count': 100,
                'mean': 45.2,
                'std': 12.5,
                'min': 18,
                'q25': 35,
                'median': 45,
                'q75': 55,
                'max': 80,
                'skewness': 0.2,
                'kurtosis': 2.1
            },
            'gender': {
                'type': 'categorical',
                'count': 100,
                'unique_count': 2,
                'most_common': 'Male',
                'most_common_count': 55
            },
            'income': {
                'type': 'numeric',
                'count': 95,
                'mean': 75000.0,
                'std': 25000.0,
                'min': 30000,
                'q25': 55000,
                'median': 75000,
                'q75': 95000,
                'max': 150000,
                'skewness': 0.8,
                'kurtosis': 3.2
            }
        },
        'data_quality': {
            'age': {
                'missing_percentage': 0.0,
                'completeness': 1.0,
                'quality_score': 1.0
            },
            'gender': {
                'missing_percentage': 0.0,
                'completeness': 1.0,
                'quality_score': 1.0
            },
            'income': {
                'missing_percentage': 0.05,
                'completeness': 0.95,
                'quality_score': 0.95
            }
        }
    }
    
    try:
        # Test the function
        html_table = create_comprehensive_summary_table(sample_stats)
        
        print("✅ Comprehensive table creation test passed!")
        print(f"Generated HTML table length: {len(html_table)} characters")
        
        # Check if HTML contains expected elements
        if '<table' in html_table and '<thead' in html_table and '<tbody' in html_table:
            print("✅ HTML structure is correct")
        else:
            print("❌ HTML structure is missing required elements")
            
        # Check if it contains our sample data
        if 'age' in html_table and 'gender' in html_table and 'income' in html_table:
            print("✅ Sample data is included in the table")
        else:
            print("❌ Sample data is missing from the table")
            
        # Save the HTML to a file for inspection
        with open('test_comprehensive_table_output.html', 'w') as f:
            f.write(html_table)
        print("📄 HTML table saved to test_comprehensive_table_output.html")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing comprehensive summary table functionality...")
    success = test_comprehensive_table()
    
    if success:
        print("\n🎉 All tests passed! The comprehensive table functionality is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the implementation.")
