import hashlib
import json
from django.core.cache import cache
from django.conf import settings

class CacheManager:
    def __init__(self):
        self.default_timeout = settings.LLM_CACHE_TIMEOUT
    
    def generate_cache_key(self, prefix, data):
        data_str = json.dumps(data, sort_keys=True)
        return f"{prefix}:{hashlib.md5(data_str.encode()).hexdigest()}"
    
    def get_cached_response(self, cache_key):
        return cache.get(cache_key)
    
    def set_cached_response(self, cache_key, data, timeout=None):
        timeout = timeout or self.default_timeout
        cache.set(cache_key, data, timeout)
    
    def cache_tool_results(self, tool_name, dataset_id, parameters, results):
        cache_key = self.generate_cache_key(f"tool:{tool_name}", {
            'dataset_id': dataset_id,
            'parameters': parameters
        })
        cache.set(cache_key, results, settings.TOOL_RESULTS_CACHE_TIMEOUT)
        return cache_key
    
    def get_cached_tool_results(self, tool_name, dataset_id, parameters):
        cache_key = self.generate_cache_key(f"tool:{tool_name}", {
            'dataset_id': dataset_id,
            'parameters': parameters
        })
        return cache.get(cache_key)
