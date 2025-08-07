import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from .cache_manager import CacheManager

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.1,
            max_tokens=4000
        )
        self.cache_manager = CacheManager()
    
    def chat(self, message, context=None):
        messages = []
        system_prompt = self._build_system_prompt(context)
        messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=message))
        response = self.model.invoke(messages)
        return response.content
    
    def interpret_summary_statistics(self, dataset_id, summary_results):
        from ..models import UserDataset
        
        try:
            dataset = UserDataset.objects.get(id=dataset_id)
            context = {
                'dataset_name': dataset.name,
                'dataset_rows': dataset.rows,
                'dataset_columns': dataset.columns,
                'summary_results': summary_results
            }
            
            prompt = f"""
            Please interpret the following summary statistics for dataset '{dataset.name}' 
            with {dataset.rows} rows and {dataset.columns} columns:
            
            {summary_results}
            
            Provide insights about:
            1. Data quality and completeness
            2. Key patterns in the data
            3. Potential issues or anomalies
            4. Recommendations for further analysis
            """
            
            return self.chat(prompt, context)
        
        except UserDataset.DoesNotExist:
            return "Error: Dataset not found"
    
    def _build_system_prompt(self, context=None):
        base_prompt = """You are an AI analytics assistant for DataFlow Analytics. 
        You can help users analyze their datasets using various statistical tools.
        
        IMPORTANT INSTRUCTIONS FOR DATA INTERPRETATION:
        1. When interpreting summary statistics, use the ACTUAL data provided in the context
        2. Format tables using proper markdown table syntax with headers and aligned columns
        3. Present data in a clear, structured format
        4. Provide specific insights based on the actual numbers, not hypothetical data
        5. Use the exact variable names and values from the provided data
        6. Format tables like this:
           | Variable | Count | Mean | Std Dev | Min | Max | 25th % | Median | 75th % |
           |----------|-------|------|---------|-----|-----|---------|--------|---------|
           | Age      | 100   | 55.2 | 10.8    | 25  | 80  | 48      | 55     | 62      |
        
        Available tools:
        - summary_statistics: Generate comprehensive summary statistics
        - correlation: Perform correlation analysis
        - regression: Perform regression analysis
        - clustering: Perform clustering analysis
        - time_series: Perform time series analysis
        - outlier_detection: Detect outliers in data
        - hypothesis_testing: Perform statistical hypothesis tests
        - data_quality: Assess data quality metrics
        - visualization: Create data visualizations
        
        When a user asks for analysis, you can:
        1. Suggest appropriate tools
        2. Execute tools and interpret results
        3. Provide insights and recommendations
        4. Guide users through the analysis process
        
        Always be helpful, clear, and provide actionable insights based on REAL data."""
        
        if context and context.get('summary_statistics'):
            summary_data = context['summary_statistics']
            dataset_name = context.get('dataset_name', 'the dataset')
            
            base_prompt += f"""
            
            CURRENT DATASET CONTEXT:
            Dataset: {dataset_name}
            Rows: {context.get('dataset_info', {}).get('rows', 'N/A')}
            Columns: {context.get('dataset_info', {}).get('columns', 'N/A')}
            
            SUMMARY STATISTICS DATA:
            {self._format_summary_statistics(summary_data)}
            
            Use this ACTUAL data for your analysis and responses. Do not generate hypothetical data."""
        
        return base_prompt
    
    def _format_summary_statistics(self, summary_data):
        """Format summary statistics for the prompt"""
        formatted = ""
        
        if 'variable_summary' in summary_data:
            formatted += "\nVARIABLE SUMMARY:\n"
            for var_name, stats in summary_data['variable_summary'].items():
                try:
                    if stats['type'] == 'numeric':
                        formatted += f"\n{var_name} (Numeric):\n"
                        formatted += f"  Count: {stats['count']}\n"
                        
                        # Handle None values for numeric stats
                        if stats['mean'] is not None:
                            formatted += f"  Mean: {stats['mean']:.2f}\n"
                        else:
                            formatted += f"  Mean: N/A\n"
                            
                        if stats['std'] is not None:
                            formatted += f"  Std Dev: {stats['std']:.2f}\n"
                        else:
                            formatted += f"  Std Dev: N/A\n"
                            
                        if stats['min'] is not None:
                            formatted += f"  Min: {stats['min']:.2f}\n"
                        else:
                            formatted += f"  Min: N/A\n"
                            
                        if stats['q25'] is not None:
                            formatted += f"  Q25: {stats['q25']:.2f}\n"
                        else:
                            formatted += f"  Q25: N/A\n"
                            
                        if stats['median'] is not None:
                            formatted += f"  Median: {stats['median']:.2f}\n"
                        else:
                            formatted += f"  Median: N/A\n"
                            
                        if stats['q75'] is not None:
                            formatted += f"  Q75: {stats['q75']:.2f}\n"
                        else:
                            formatted += f"  Q75: N/A\n"
                            
                        if stats['max'] is not None:
                            formatted += f"  Max: {stats['max']:.2f}\n"
                        else:
                            formatted += f"  Max: N/A\n"
                            
                        if stats['skewness'] is not None:
                            formatted += f"  Skewness: {stats['skewness']:.2f}\n"
                        else:
                            formatted += f"  Skewness: N/A\n"
                            
                        if stats['kurtosis'] is not None:
                            formatted += f"  Kurtosis: {stats['kurtosis']:.2f}\n"
                        else:
                            formatted += f"  Kurtosis: N/A\n"
                            
                    elif stats['type'] == 'categorical':
                        formatted += f"\n{var_name} (Categorical):\n"
                        formatted += f"  Count: {stats['count']}\n"
                        formatted += f"  Unique Values: {stats['unique_count']}\n"
                        formatted += f"  Most Common: {stats['most_common']}\n"
                        formatted += f"  Most Common Count: {stats['most_common_count']}\n"
                    else:
                        formatted += f"\n{var_name} (Unknown Type):\n"
                        formatted += f"  Count: {stats['count']}\n"
                        if 'error' in stats:
                            formatted += f"  Error: {stats['error']}\n"
                except Exception as e:
                    formatted += f"\n{var_name} (Error processing): {str(e)}\n"
        
        if 'data_quality' in summary_data:
            formatted += "\nDATA QUALITY:\n"
            for var_name, quality in summary_data['data_quality'].items():
                try:
                    formatted += f"\n{var_name}:\n"
                    formatted += f"  Missing %: {quality['missing_percentage']:.2%}\n"
                    formatted += f"  Completeness: {quality['completeness']:.2%}\n"
                    formatted += f"  Quality Score: {quality['quality_score']:.2f}\n"
                except Exception as e:
                    formatted += f"\n{var_name} (Error processing quality): {str(e)}\n"
        
        if 'correlation_matrix' in summary_data and summary_data['correlation_matrix'].get('strong_correlations'):
            formatted += "\nSTRONG CORRELATIONS:\n"
            for corr in summary_data['correlation_matrix']['strong_correlations']:
                try:
                    formatted += f"  {corr['variable1']} ↔ {corr['variable2']}: {corr['correlation']:.3f}\n"
                except Exception as e:
                    formatted += f"  Error processing correlation: {str(e)}\n"
        
        return formatted
