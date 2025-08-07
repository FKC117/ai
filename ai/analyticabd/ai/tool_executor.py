import json
import logging
from typing import Dict, Any, Optional, List
from django.conf import settings
from .cache_manager import CacheManager
from .conversation_manager import ConversationManager
from .llm_client import LLMClient
from ..tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

class ToolExecutor:
    def __init__(self):
        self.cache_manager = CacheManager()
        self.tool_registry = ToolRegistry()
        self.llm_client = LLMClient()
    
    def execute_tool(self, tool_name: str, dataset_id: int, user_id: int, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a specific tool with given parameters
        
        Args:
            tool_name: Name of the tool to execute
            dataset_id: ID of the dataset to analyze
            user_id: ID of the user requesting the analysis
            parameters: Tool-specific parameters
            
        Returns:
            Tool execution results
        """
        try:
            # Get tool from registry
            tool = self.tool_registry.get_tool(tool_name, dataset_id, user_id)
            
            if not tool:
                return {
                    'error': f'Tool "{tool_name}" not found',
                    'available_tools': self.tool_registry.list_available_tools()
                }
            
            # Check cache for existing results
            cached_results = self.cache_manager.get_cached_tool_results(
                tool_name, dataset_id, parameters or {}
            )
            
            if cached_results:
                logger.info(f"Returning cached results for tool {tool_name}")
                return {
                    'success': True,
                    'cached': True,
                    'results': cached_results,
                    'tool_name': tool_name,
                    'dataset_id': dataset_id
                }
            
            # Execute tool
            logger.info(f"Executing tool {tool_name} for dataset {dataset_id}")
            results = tool.execute(parameters)
            
            # Cache results
            self.cache_manager.cache_tool_results(
                tool_name, dataset_id, parameters or {}, results
            )
            
            return {
                'success': True,
                'cached': False,
                'results': results,
                'tool_name': tool_name,
                'dataset_id': dataset_id
            }
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {
                'error': f'Error executing tool "{tool_name}": {str(e)}',
                'tool_name': tool_name,
                'dataset_id': dataset_id
            }
    
    def process_chat_message(self, user_id: int, message: str, session_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a chat message and determine appropriate tool execution
        
        Args:
            user_id: ID of the user
            message: User's message
            session_id: Chat session ID
            context: Additional context
            
        Returns:
            Response with tool execution results and LLM interpretation
        """
        try:
            # Initialize conversation manager
            conversation_manager = ConversationManager(user_id, session_id)
            conversation_manager.add_message('user', message, context)
            
            # Analyze message to determine if tool execution is needed
            tool_request = self._analyze_message_for_tools(message, context)
            
            if tool_request:
                # Execute the requested tool
                tool_results = self.execute_tool(
                    tool_request['tool_name'],
                    tool_request['dataset_id'],
                    user_id,
                    tool_request['parameters']
                )
                
                # Add tool execution to conversation
                if tool_results.get('success'):
                    conversation_manager.add_tool_execution(
                        tool_request['tool_name'],
                        tool_request['parameters'],
                        tool_results['results']
                    )
                
                # Generate LLM interpretation
                interpretation = self._generate_tool_interpretation(
                    tool_request['tool_name'],
                    tool_results,
                    message
                )
                
                conversation_manager.add_message('assistant', interpretation)
                
                return {
                    'response': interpretation,
                    'tool_executed': tool_request['tool_name'],
                    'tool_results': tool_results,
                    'metadata': {
                        'session_id': session_id,
                        'user_id': user_id
                    }
                }
            else:
                # No tool execution needed, generate general response
                response = self.llm_client.chat(message, context)
                conversation_manager.add_message('assistant', response)
                
                return {
                    'response': response,
                    'tool_executed': None,
                    'metadata': {
                        'session_id': session_id,
                        'user_id': user_id
                    }
                }
                
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            return {
                'error': f'Error processing message: {str(e)}',
                'response': 'I encountered an error while processing your request. Please try again.'
            }
    
    def _analyze_message_for_tools(self, message: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Analyze user message to determine if tool execution is needed
        
        Args:
            message: User's message
            context: Additional context
            
        Returns:
            Tool request dictionary or None
        """
        message_lower = message.lower()
        
        # Simple keyword-based tool detection (can be enhanced with LLM)
        tool_mappings = {
            'summary': 'summary_statistics',
            'statistics': 'summary_statistics',
            'describe': 'summary_statistics',
            'correlation': 'correlation',
            'correlate': 'correlation',
            'regression': 'regression',
            'predict': 'regression',
            'cluster': 'clustering',
            'group': 'clustering',
            'outlier': 'outlier_detection',
            'anomaly': 'outlier_detection',
            'quality': 'data_quality',
            'missing': 'data_quality',
            'visualize': 'visualization',
            'plot': 'visualization',
            'chart': 'visualization',
            'hypothesis': 'hypothesis_testing',
            'test': 'hypothesis_testing',
            'time series': 'time_series',
            'trend': 'time_series'
        }
        
        # Check for tool keywords
        for keyword, tool_name in tool_mappings.items():
            if keyword in message_lower:
                # Extract dataset ID from context
                dataset_id = context.get('dataset_id') if context else None
                
                if dataset_id:
                    return {
                        'tool_name': tool_name,
                        'dataset_id': dataset_id,
                        'parameters': self._extract_parameters(message, tool_name)
                    }
        
        return None
    
    def _extract_parameters(self, message: str, tool_name: str) -> Dict[str, Any]:
        """
        Extract tool-specific parameters from user message
        
        Args:
            message: User's message
            tool_name: Name of the tool
            
        Returns:
            Dictionary of parameters
        """
        parameters = {}
        
        # Tool-specific parameter extraction
        if tool_name == 'correlation':
            # Extract correlation type
            if 'spearman' in message.lower():
                parameters['correlation_type'] = 'spearman'
            elif 'kendall' in message.lower():
                parameters['correlation_type'] = 'kendall'
            else:
                parameters['correlation_type'] = 'pearson'
        
        elif tool_name == 'regression':
            # Extract target and independent variables (simplified)
            # This could be enhanced with NLP parsing
            parameters['regression_type'] = 'linear'
        
        elif tool_name == 'outlier_detection':
            # Extract outlier threshold
            if 'strict' in message.lower():
                parameters['outlier_threshold'] = 1.0
            elif 'lenient' in message.lower():
                parameters['outlier_threshold'] = 2.0
            else:
                parameters['outlier_threshold'] = 1.5
        
        return parameters
    
    def _generate_tool_interpretation(self, tool_name: str, tool_results: Dict[str, Any], original_message: str) -> str:
        """
        Generate LLM interpretation of tool results
        
        Args:
            tool_name: Name of the tool executed
            tool_results: Results from tool execution
            original_message: Original user message
            
        Returns:
            LLM-generated interpretation
        """
        if not tool_results.get('success'):
            return f"I encountered an error while executing the {tool_name} analysis: {tool_results.get('error', 'Unknown error')}"
        
        # Create interpretation prompt
        interpretation_prompt = f"""
        The user asked: "{original_message}"
        
        I executed the {tool_name} tool and got the following results:
        {json.dumps(tool_results.get('results', {}), indent=2)}
        
        Please provide a clear, helpful interpretation of these results that the user can understand.
        Focus on key insights and actionable recommendations.
        """
        
        return self.llm_client.chat(interpretation_prompt)
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools with descriptions
        
        Returns:
            List of tool information
        """
        tools = []
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        
        for tool_name, description in tool_descriptions.items():
            tools.append({
                'name': tool_name,
                'description': description,
                'parameters_schema': self._get_tool_parameters_schema(tool_name)
            })
        
        return tools
    
    def _get_tool_parameters_schema(self, tool_name: str) -> Dict[str, Any]:
        """
        Get parameter schema for a specific tool
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Parameter schema dictionary
        """
        try:
            tool = self.tool_registry.get_tool(tool_name, 0, 0)
            if tool:
                return tool.get_parameters_schema()
        except Exception as e:
            logger.warning(f"Could not get parameters schema for tool {tool_name}: {str(e)}")
        
        return {}
    
    def batch_execute_tools(self, tool_requests: List[Dict[str, Any]], user_id: int) -> List[Dict[str, Any]]:
        """
        Execute multiple tools in batch
        
        Args:
            tool_requests: List of tool execution requests
            user_id: ID of the user
            
        Returns:
            List of tool execution results
        """
        results = []
        
        for request in tool_requests:
            result = self.execute_tool(
                request['tool_name'],
                request['dataset_id'],
                user_id,
                request.get('parameters', {})
            )
            results.append(result)
        
        return results
