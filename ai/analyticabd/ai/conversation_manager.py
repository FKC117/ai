import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from django.core.cache import cache
from .cache_manager import CacheManager

class ConversationManager:
    def __init__(self, user_id: int, session_id: Optional[str] = None):
        self.user_id = user_id
        self.session_id = session_id or f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history = []
        self.current_context = {}
        self.cache_manager = CacheManager()
        self.max_history_length = 50  # Maximum number of messages to keep in memory
        
        # Load existing conversation if session_id is provided
        if session_id:
            self.load_conversation()
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to the conversation history
        
        Args:
            role: 'user', 'assistant', or 'system'
            content: Message content
            metadata: Additional metadata for the message
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.conversation_history.append(message)
        
        # Keep only the last max_history_length messages
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        # Update cache
        self._save_conversation()
    
    def get_context(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """
        Return conversation context for LLM
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            List of recent messages formatted for LLM
        """
        recent_messages = self.conversation_history[-max_messages:] if len(self.conversation_history) > max_messages else self.conversation_history
        
        context = []
        for message in recent_messages:
            context.append({
                'role': message['role'],
                'content': message['content']
            })
        
        return context
    
    def get_full_context(self) -> Dict[str, Any]:
        """
        Get full conversation context including metadata
        
        Returns:
            Dictionary with conversation context and metadata
        """
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'conversation_history': self.conversation_history,
            'current_context': self.current_context,
            'message_count': len(self.conversation_history)
        }
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        self.current_context = {}
        self._save_conversation()
    
    def update_context(self, key: str, value: Any) -> None:
        """
        Update the current context with new information
        
        Args:
            key: Context key
            value: Context value
        """
        self.current_context[key] = value
        self._save_conversation()
    
    def get_context_value(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the current context
        
        Args:
            key: Context key
            default: Default value if key doesn't exist
            
        Returns:
            Context value or default
        """
        return self.current_context.get(key, default)
    
    def add_tool_execution(self, tool_name: str, parameters: Dict[str, Any], results: Dict[str, Any]) -> None:
        """
        Add a tool execution to the conversation context
        
        Args:
            tool_name: Name of the tool executed
            parameters: Tool parameters
            results: Tool execution results
        """
        execution_info = {
            'tool_name': tool_name,
            'parameters': parameters,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.current_context['last_tool_execution'] = execution_info
        self.add_message('system', f"Tool '{tool_name}' executed with parameters: {parameters}")
    
    def get_last_tool_execution(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the last tool execution
        
        Returns:
            Last tool execution info or None
        """
        return self.current_context.get('last_tool_execution')
    
    def add_dataset_context(self, dataset_id: int, dataset_name: str) -> None:
        """
        Add dataset context to the conversation
        
        Args:
            dataset_id: Dataset ID
            dataset_name: Dataset name
        """
        self.current_context['current_dataset'] = {
            'id': dataset_id,
            'name': dataset_name,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_current_dataset(self) -> Optional[Dict[str, Any]]:
        """
        Get current dataset information
        
        Returns:
            Current dataset info or None
        """
        return self.current_context.get('current_dataset')
    
    def _save_conversation(self) -> None:
        """Save conversation to cache"""
        cache_key = f"conversation:{self.session_id}"
        conversation_data = {
            'user_id': self.user_id,
            'session_id': self.session_id,
            'conversation_history': self.conversation_history,
            'current_context': self.current_context,
            'last_updated': datetime.now().isoformat()
        }
        
        self.cache_manager.set_cached_response(
            cache_key, 
            conversation_data, 
            timeout=3600  # 1 hour timeout
        )
    
    def load_conversation(self) -> bool:
        """
        Load conversation from cache
        
        Returns:
            True if conversation was loaded, False otherwise
        """
        cache_key = f"conversation:{self.session_id}"
        conversation_data = self.cache_manager.get_cached_response(cache_key)
        
        if conversation_data:
            self.conversation_history = conversation_data.get('conversation_history', [])
            self.current_context = conversation_data.get('current_context', {})
            return True
        
        return False
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation
        
        Returns:
            Conversation summary
        """
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'message_count': len(self.conversation_history),
            'current_dataset': self.get_current_dataset(),
            'last_tool_execution': self.get_last_tool_execution(),
            'conversation_duration': self._calculate_duration()
        }
    
    def _calculate_duration(self) -> Optional[str]:
        """Calculate conversation duration"""
        if len(self.conversation_history) < 2:
            return None
        
        first_message = self.conversation_history[0]
        last_message = self.conversation_history[-1]
        
        try:
            first_time = datetime.fromisoformat(first_message['timestamp'])
            last_time = datetime.fromisoformat(last_message['timestamp'])
            duration = last_time - first_time
            return str(duration)
        except (ValueError, KeyError):
            return None
    
    def export_conversation(self) -> Dict[str, Any]:
        """
        Export conversation for storage or analysis
        
        Returns:
            Complete conversation data
        """
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'conversation_history': self.conversation_history,
            'current_context': self.current_context,
            'export_timestamp': datetime.now().isoformat(),
            'summary': self.get_conversation_summary()
        }
