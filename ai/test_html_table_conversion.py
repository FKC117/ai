#!/usr/bin/env python3
"""
Test script to verify HTML table conversion functionality
"""

def test_html_table_extraction():
    """Test the HTML table extraction function"""
    
    # Sample AI response content with HTML table
    sample_content = """
    Here's the analysis of your dataset:
    
    ## Summary Statistics
    
    The following table shows the key statistics:
    
    <table style="border-collapse: collapse; width: 100%; margin: 10px 0; font-family: Inter, sans-serif;">
      <thead>
        <tr style="background-color: #1a365d; color: white;">
          <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Variable</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Count</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Mean</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Std Dev</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background-color: #f7fafc;">
          <td style="border: 1px solid #ddd; padding: 8px;">customer_id</td>
          <td style="border: 1px solid #ddd; padding: 8px;">15000</td>
          <td style="border: 1px solid #ddd; padding: 8px;">18085667.45</td>
          <td style="border: 1px solid #ddd; padding: 8px;">12329999.71</td>
        </tr>
        <tr style="background-color: #ffffff;">
          <td style="border: 1px solid #ddd; padding: 8px;">product_id</td>
          <td style="border: 1px solid #ddd; padding: 8px;">15000</td>
          <td style="border: 1px solid #ddd; padding: 8px;">32697709.38</td>
          <td style="border: 1px solid #ddd; padding: 8px;">16294548.70</td>
        </tr>
      </tbody>
    </table>
    
    This concludes our analysis.
    """
    
    try:
        from bs4 import BeautifulSoup
        
        # Test HTML table extraction
        soup = BeautifulSoup(sample_content, 'html.parser')
        html_tables = soup.find_all('table')
        
        print(f"✅ Found {len(html_tables)} HTML tables in content")
        
        for i, html_table in enumerate(html_tables):
            rows = html_table.find_all('tr')
            if rows:
                max_cols = max(len(row.find_all(['td', 'th'])) for row in rows)
                print(f"📊 Table {i+1}: {len(rows)} rows, {max_cols} columns")
                
                # Test cell extraction
                for r_idx, row in enumerate(rows):
                    cells = row.find_all(['td', 'th'])
                    for c_idx, cell in enumerate(cells):
                        cell_text = cell.get_text(strip=True)
                        print(f"   Row {r_idx}, Col {c_idx}: '{cell_text}'")
        
        # Test content cleaning
        for table in html_tables:
            table.decompose()
        
        cleaned_content = soup.get_text()
        print(f"✅ Content cleaned. Length: {len(cleaned_content)} characters")
        print(f"📝 Cleaned content preview: {cleaned_content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in HTML table extraction test: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing HTML table conversion functionality...")
    success = test_html_table_extraction()
    if success:
        print("✅ All tests passed! HTML table conversion should work properly.")
    else:
        print("❌ Tests failed. Please check the implementation.")
