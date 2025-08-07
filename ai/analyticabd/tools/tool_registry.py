class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self._register_all_tools()
    
    def _register_all_tools(self):
        from .summary_statistics_tool import SummaryStatisticsTool
        
        # Register available tools
        self.tools.update({
            'summary_statistics': SummaryStatisticsTool,
        })
        
        # Future tools will be added here:
        # 'correlation': CorrelationTool,
        # 'regression': RegressionTool,
        # 'clustering': ClusteringTool,
        # 'time_series': TimeSeriesTool,
        # 'outlier_detection': OutlierDetectionTool,
        # 'hypothesis_testing': HypothesisTestingTool,
        # 'data_quality': DataQualityTool,
        # 'visualization': VisualizationTool,
    
    def get_tool(self, tool_name, dataset_id, user_id):
        if tool_name in self.tools:
            return self.tools[tool_name](dataset_id, user_id)
        return None
    
    def list_available_tools(self):
        return list(self.tools.keys())
    
    def get_tool_descriptions(self):
        descriptions = {}
        for tool_name, tool_class in self.tools.items():
            temp_tool = tool_class(0, 0)
            descriptions[tool_name] = temp_tool.get_description()
        return descriptions
    
    def register_tool(self, tool_name, tool_class):
        """Register a new tool dynamically"""
        self.tools[tool_name] = tool_class
    
    def unregister_tool(self, tool_name):
        """Unregister a tool"""
        if tool_name in self.tools:
            del self.tools[tool_name]
    
    def get_tool_info(self, tool_name):
        """Get detailed information about a tool"""
        if tool_name not in self.tools:
            return None
        
        tool_class = self.tools[tool_name]
        temp_tool = tool_class(0, 0)
        
        return {
            'name': tool_name,
            'description': temp_tool.get_description(),
            'parameters_schema': temp_tool.get_parameters_schema(),
            'class_name': tool_class.__name__
        }
