#!/usr/bin/env python3
"""
Test script to verify the new AI response processing logic
"""

from bs4 import BeautifulSoup, NavigableString

def is_table_line(line):
    """Check if a line is part of a markdown table"""
    line_stripped = line.strip()
    # Check for pipe-separated tables
    if line_stripped.startswith('|') and line_stripped.endswith('|'):
        return True
    # Check for separator rows (dashes)
    if line_stripped.replace('-', '').replace(':', '').replace('|', '').strip() == '' and '|' in line_stripped:
        return True
    # Check for tab-separated tables (common in AI responses)
    if '\t' in line_stripped and len(line_stripped.split('\t')) > 1:
        return True
    return False

def process_ai_response_content(content):
    """Simulate the new AI response processing logic"""
    print("=== Processing AI Response Content ===")
    print(f"Original content length: {len(content)}")
    
    # Parse content with BeautifulSoup to preserve exact order
    soup = BeautifulSoup(content, 'html.parser')
    processed_elements = []
    
    # Process each top-level element in order
    for element in soup.contents:
        if element.name == 'table':
            # This is an HTML table
            processed_elements.append(f"[HTML TABLE: {len(str(element))} chars]")
            print(f"✅ Found HTML table: {len(str(element))} chars")
        elif isinstance(element, NavigableString):
            # This is plain text - process it line by line
            text_content = str(element)
            lines = text_content.splitlines()
            current_paragraph = []
            
            for line in lines:
                line_stripped = line.strip()
                
                # Check if this line is part of a markdown table
                if is_table_line(line):
                    # Process any accumulated text first
                    if current_paragraph:
                        paragraph_text = '\n'.join(current_paragraph).strip()
                        if paragraph_text:
                            processed_elements.append(f"[TEXT: {paragraph_text[:50]}...]")
                            print(f"📝 Processed text: {paragraph_text[:50]}...")
                        current_paragraph = []
                    
                    processed_elements.append(f"[MARKDOWN TABLE LINE: {line[:50]}...]")
                    print(f"📊 Found markdown table line: {line[:50]}...")
                else:
                    # If this is a blank line, process accumulated paragraph
                    if not line_stripped and current_paragraph:
                        paragraph_text = '\n'.join(current_paragraph).strip()
                        if paragraph_text:
                            processed_elements.append(f"[TEXT: {paragraph_text[:50]}...]")
                            print(f"📝 Processed text: {paragraph_text[:50]}...")
                        current_paragraph = []
                    else:
                        # Add line to current paragraph
                        current_paragraph.append(line)
            
            # Process any remaining paragraph
            if current_paragraph:
                paragraph_text = '\n'.join(current_paragraph).strip()
                if paragraph_text:
                    processed_elements.append(f"[TEXT: {paragraph_text[:50]}...]")
                    print(f"📝 Processed text: {paragraph_text[:50]}...")
        else:
            # Handle other HTML elements (if any) as text
            text_content = element.get_text()
            if text_content.strip():
                processed_elements.append(f"[OTHER HTML: {text_content[:50]}...]")
                print(f"🔧 Processed other HTML: {text_content[:50]}...")
    
    print(f"\n=== Processing Summary ===")
    print(f"Total elements processed: {len(processed_elements)}")
    for i, element in enumerate(processed_elements):
        print(f"{i+1}. {element}")
    
    return processed_elements

# Test with the user's example
test_content = """Okay, let's outline the next steps for analyzing the customer_details.csv dataset. Since we have summary statistics already, we can move towards more insightful analysis. The following table suggests potential next steps, categorized by analytical goal, along with the recommended tools and expected outputs.

Analytical Goal	Recommended Tool	Expected Output	Actionable Insights
Understand Customer Segmentation	Clustering (e.g., K-means)	Customer segments based on demographics, purchase behavior, and preferences. Visualization: PNG image (800x600px), scatter plot showing clusters, colored by segment.	Identify distinct customer groups to tailor marketing strategies.
Analyze Purchase Behavior	Correlation & Regression	Correlation matrix showing relationships between variables (e.g., Age, Purchase Amount, Previous Purchases). Regression model predicting Purchase Amount based on other variables. Visualization: PNG image (800x600px), heatmap of correlation matrix.	Identify factors influencing purchase amount and customer lifetime value (CLTV).
Evaluate the Effectiveness of Promotions	Hypothesis Testing (e.g., t-test, ANOVA)	Statistical significance of differences in purchase amounts between groups with and without promo codes.	Determine if promotional campaigns are effective in driving sales.
Visualize Key Metrics	Visualization (Bar charts, Histograms)	PNG images (800x600px each): Histogram of Age, Bar chart of Purchase Amount by Category, Bar chart of Review Ratings. Use a consistent color scheme (e.g., blues and greens).	Gain a clear understanding of the distribution of key variables and identify potential areas for improvement.
Assess Data Quality (Further)	Data Quality Assessment	Detailed report on data completeness, consistency, and accuracy for each variable.	Identify and address any data quality issues that might affect the analysis.

This table provides a structured approach to further analysis. Remember to consider the business questions you want to answer when prioritizing these steps. Let me know which step you'd like to start with!"""

print("Testing AI response processing...")
result = process_ai_response_content(test_content)

print("\n=== Expected vs Actual ===")
print("Expected: Text -> Markdown Table -> Text")
print("Actual:", " -> ".join([elem.split(':')[0] for elem in result]))
