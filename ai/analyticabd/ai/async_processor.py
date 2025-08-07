import logging
from typing import Dict, Any, List, Optional
from celery import shared_task
from django.conf import settings
from .tool_executor import ToolExecutor
from .cache_manager import CacheManager
from .conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def execute_tool_async(self, tool_name: str, dataset_id: int, user_id: int, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Async task to execute a tool
    
    Args:
        tool_name: Name of the tool to execute
        dataset_id: ID of the dataset to analyze
        user_id: ID of the user requesting the analysis
        parameters: Tool-specific parameters
        
    Returns:
        Tool execution results
    """
    try:
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Executing tool', 'tool_name': tool_name}
        )
        
        # Execute tool
        executor = ToolExecutor()
        results = executor.execute_tool(tool_name, dataset_id, user_id, parameters)
        
        # Update task state with results
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'Tool execution completed',
                'tool_name': tool_name,
                'results': results
            }
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in async tool execution: {str(e)}")
        
        # Update task state with error
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'Tool execution failed',
                'error': str(e),
                'tool_name': tool_name
            }
        )
        
        return {
            'error': f'Async tool execution failed: {str(e)}',
            'tool_name': tool_name,
            'dataset_id': dataset_id
        }

@shared_task(bind=True)
def batch_execute_tools_async(self, tool_requests: List[Dict[str, Any]], user_id: int) -> Dict[str, Any]:
    """
    Async task to execute multiple tools in batch
    
    Args:
        tool_requests: List of tool execution requests
        user_id: ID of the user
        
    Returns:
        Batch execution results
    """
    try:
        total_tools = len(tool_requests)
        completed_tools = 0
        results = []
        
        # Update initial state
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'Starting batch execution',
                'total_tools': total_tools,
                'completed_tools': completed_tools
            }
        )
        
        executor = ToolExecutor()
        
        for i, request in enumerate(tool_requests):
            # Update progress
            completed_tools = i + 1
            self.update_state(
                state='PROGRESS',
                meta={
                    'status': f'Executing tool {completed_tools}/{total_tools}',
                    'current_tool': request.get('tool_name'),
                    'total_tools': total_tools,
                    'completed_tools': completed_tools
                }
            )
            
            # Execute tool
            result = executor.execute_tool(
                request['tool_name'],
                request['dataset_id'],
                user_id,
                request.get('parameters', {})
            )
            
            results.append(result)
        
        # Update final state
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'Batch execution completed',
                'total_tools': total_tools,
                'completed_tools': completed_tools,
                'results': results
            }
        )
        
        return {
            'success': True,
            'total_tools': total_tools,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error in batch tool execution: {str(e)}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'Batch execution failed',
                'error': str(e)
            }
        )
        
        return {
            'error': f'Batch tool execution failed: {str(e)}',
            'user_id': user_id
        }

@shared_task(bind=True)
def process_chat_message_async(self, user_id: int, message: str, session_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Async task to process chat message
    
    Args:
        user_id: ID of the user
        message: User's message
        session_id: Chat session ID
        context: Additional context
        
    Returns:
        Chat processing results
    """
    try:
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Processing chat message'}
        )
        
        # Process message
        executor = ToolExecutor()
        results = executor.process_chat_message(user_id, message, session_id, context)
        
        # Update task state with results
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'Chat message processed',
                'results': results
            }
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in async chat processing: {str(e)}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'Chat processing failed',
                'error': str(e)
            }
        )
        
        return {
            'error': f'Async chat processing failed: {str(e)}',
            'response': 'I encountered an error while processing your request. Please try again.'
        }

@shared_task(bind=True)
def cache_cleanup_async(self) -> Dict[str, Any]:
    """
    Async task to clean up expired cache entries
    
    Returns:
        Cleanup results
    """
    try:
        cache_manager = CacheManager()
        
        # This is a placeholder for cache cleanup logic
        # In a real implementation, you would iterate through cache keys
        # and remove expired entries
        
        self.update_state(
            state='SUCCESS',
            meta={'status': 'Cache cleanup completed'}
        )
        
        return {
            'success': True,
            'status': 'Cache cleanup completed'
        }
        
    except Exception as e:
        logger.error(f"Error in cache cleanup: {str(e)}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'Cache cleanup failed',
                'error': str(e)
            }
        )
        
        return {
            'error': f'Cache cleanup failed: {str(e)}'
        }

@shared_task(bind=True)
def generate_report_async(self, dataset_id: int, user_id: int, report_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Async task to generate comprehensive reports
    
    Args:
        dataset_id: ID of the dataset
        user_id: ID of the user
        report_type: Type of report to generate
        parameters: Report parameters
        
    Returns:
        Report generation results
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': f'Generating {report_type} report'}
        )
        
        executor = ToolExecutor()
        
        # Define report tools based on report type
        report_tools = {
            'comprehensive': ['summary_statistics', 'correlation', 'data_quality', 'outlier_detection'],
            'data_quality': ['data_quality', 'summary_statistics'],
            'statistical': ['summary_statistics', 'correlation', 'hypothesis_testing'],
            'visualization': ['summary_statistics', 'visualization'],
            'custom': parameters.get('tools', []) if parameters else []
        }
        
        tools_to_execute = report_tools.get(report_type, ['summary_statistics'])
        
        # Execute tools for report
        tool_requests = [
            {
                'tool_name': tool_name,
                'dataset_id': dataset_id,
                'parameters': parameters.get('tool_parameters', {}).get(tool_name, {})
            }
            for tool_name in tools_to_execute
        ]
        
        # Execute tools
        results = []
        for request in tool_requests:
            result = executor.execute_tool(
                request['tool_name'],
                request['dataset_id'],
                user_id,
                request.get('parameters', {})
            )
            results.append(result)
        
        # Generate report summary
        report_summary = _generate_report_summary(results, report_type)
        
        self.update_state(
            state='SUCCESS',
            meta={
                'status': f'{report_type} report generated',
                'report_summary': report_summary,
                'tool_results': results
            }
        )
        
        return {
            'success': True,
            'report_type': report_type,
            'report_summary': report_summary,
            'tool_results': results
        }
        
    except Exception as e:
        logger.error(f"Error in report generation: {str(e)}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'Report generation failed',
                'error': str(e)
            }
        )
        
        return {
            'error': f'Report generation failed: {str(e)}',
            'report_type': report_type
        }

def _generate_report_summary(tool_results: List[Dict[str, Any]], report_type: str) -> str:
    """
    Generate a summary of report results
    
    Args:
        tool_results: Results from tool executions
        report_type: Type of report
        
    Returns:
        Report summary
    """
    summary_parts = []
    
    for result in tool_results:
        if result.get('success'):
            tool_name = result.get('tool_name', 'Unknown')
            summary_parts.append(f"- {tool_name}: Analysis completed successfully")
        else:
            tool_name = result.get('tool_name', 'Unknown')
            error = result.get('error', 'Unknown error')
            summary_parts.append(f"- {tool_name}: Failed - {error}")
    
    summary = f"{report_type.title()} Report Summary:\n" + "\n".join(summary_parts)
    return summary

# Task monitoring and management functions
def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of an async task
    
    Args:
        task_id: ID of the task
        
    Returns:
        Task status information
    """
    from celery.result import AsyncResult
    
    task_result = AsyncResult(task_id)
    
    return {
        'task_id': task_id,
        'status': task_result.status,
        'result': task_result.result if task_result.ready() else None,
        'meta': task_result.info if hasattr(task_result, 'info') else None
    }

def cancel_task(task_id: str) -> Dict[str, Any]:
    """
    Cancel an async task
    
    Args:
        task_id: ID of the task to cancel
        
    Returns:
        Cancellation result
    """
    from celery.result import AsyncResult
    
    task_result = AsyncResult(task_id)
    
    if task_result.state in ['PENDING', 'STARTED']:
        task_result.revoke(terminate=True)
        return {
            'success': True,
            'message': f'Task {task_id} cancelled successfully'
        }
    else:
        return {
            'success': False,
            'message': f'Cannot cancel task {task_id} in state {task_result.state}'
        }

def get_user_tasks(user_id: int) -> List[Dict[str, Any]]:
    """
    Get all tasks for a specific user
    
    Args:
        user_id: ID of the user
        
    Returns:
        List of user's tasks
    """
    # This would typically query a database table that tracks user tasks
    # For now, return empty list as placeholder
    return []

def cleanup_user_tasks(user_id: int) -> Dict[str, Any]:
    """
    Clean up completed tasks for a user
    
    Args:
        user_id: ID of the user
        
    Returns:
        Cleanup result
    """
    # This would typically remove completed tasks from tracking
    # For now, return success as placeholder
    return {
        'success': True,
        'message': f'Cleaned up tasks for user {user_id}'
    }
