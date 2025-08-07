import hashlib
import json
import logging
from django.core.cache import cache
from django.conf import settings

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        self.default_timeout = settings.LLM_CACHE_TIMEOUT
        self._cache_available = True
    
    def generate_cache_key(self, prefix, data):
        data_str = json.dumps(data, sort_keys=True)
        return f"{prefix}:{hashlib.md5(data_str.encode()).hexdigest()}"
    
    def _safe_cache_operation(self, operation, *args, **kwargs):
        """Safely execute cache operations with error handling"""
        if not self._cache_available:
            return None
        
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Cache operation failed: {str(e)}. Disabling cache for this session.")
            self._cache_available = False
            return None
    
    def get_cached_response(self, cache_key):
        return self._safe_cache_operation(cache.get, cache_key)
    
    def set_cached_response(self, cache_key, data, timeout=None):
        timeout = timeout or self.default_timeout
        return self._safe_cache_operation(cache.set, cache_key, data, timeout)
    
    def cache_tool_results(self, tool_name, dataset_id, parameters, results):
        cache_key = self.generate_cache_key(f"tool:{tool_name}", {
            'dataset_id': dataset_id,
            'parameters': parameters
        })
        return self._safe_cache_operation(cache.set, cache_key, results, settings.TOOL_RESULTS_CACHE_TIMEOUT)
    
    def get_cached_tool_results(self, tool_name, dataset_id, parameters):
        cache_key = self.generate_cache_key(f"tool:{tool_name}", {
            'dataset_id': dataset_id,
            'parameters': parameters
        })
        return self._safe_cache_operation(cache.get, cache_key)
