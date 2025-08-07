from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any
from ..ai.cache_manager import CacheManager

class BaseTool(ABC):
    def __init__(self, dataset_id, user_id):
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.dataset = None
        self.cache_manager = CacheManager()
        self.tool_name = self.__class__.__name__.lower()
    
    @abstractmethod
    def execute(self, parameters=None):
        pass
    
    @abstractmethod
    def get_description(self):
        pass
    
    @abstractmethod
    def get_parameters_schema(self):
        pass
    
    def load_dataset(self):
        from ..models import UserDataset
        cache_key = f"dataset:{self.dataset_id}"
        cached_dataset = self.cache_manager.get_cached_response(cache_key)
        
        if cached_dataset:
            self.dataset = pd.read_json(cached_dataset)
        else:
            user_dataset = UserDataset.objects.get(id=self.dataset_id)
            self.dataset = pd.read_csv(user_dataset.file.path)
            self.cache_manager.set_cached_response(cache_key, self.dataset.to_json())
