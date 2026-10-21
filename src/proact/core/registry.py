# proact/core/registry.py
from typing import Callable, Dict, TypeVar, Generic

T = TypeVar("T")

class Registry(Generic[T]):
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, T] = {}

    def register(self, key: str):
        def deco(obj: T) -> T:
            if key in self._items:
                raise KeyError(f"{self.name}: '{key}' already registered")
            self._items[key] = obj
            return obj
        return deco

    def get(self, key: str) -> T:
        if key not in self._items:
            raise KeyError(f"{self.name}: '{key}' not found. Available: {list(self._items)}")
        return self._items[key]

    def keys(self):
        return list(self._items.keys())

# Domain singletons
MODEL_REGISTRY   = Registry[Callable]("MODEL_REGISTRY")
DATASET_REGISTRY = Registry[Callable]("DATASET_REGISTRY")
COLLATE_REGISTRY = Registry[Callable]("COLLATE_REGISTRY")

WORKFLOW_FUNCTIONS_REGISTRY = Registry[Callable]("FUNCTION_STEP_REGISTRY")