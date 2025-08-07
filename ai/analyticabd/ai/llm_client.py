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
        
        Always be helpful, clear, and provide actionable insights."""
        
        if context:
            base_prompt += f"\n\nContext: {context}"
        
        return base_prompt
