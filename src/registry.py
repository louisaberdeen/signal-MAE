"""
Generic plugin registry system for models, data loaders, and transforms.

This module provides a decorator-based registration pattern that allows
components to be registered and instantiated by name.

Example:
    from src.registry import model_registry

    @model_registry.register("my-model")
    class MyModel(BaseModel):
        ...

    # Later, instantiate by name
    model = model_registry.create("my-model", config)
"""

from typing import Dict, Type, TypeVar, Callable, List, Any, Optional

T = TypeVar('T')


class PluginRegistry:
    """
    Generic registry for plugin components with decorator-based registration.

    Supports:
    - Decorator-based registration
    - Factory method instantiation
    - Plugin discovery and listing
    - Override protection (configurable)

    Attributes:
        name: Unique identifier for this registry (e.g., 'models')
    """

    # Class-level storage for all registries
    _all_registries: Dict[str, 'PluginRegistry'] = {}

    def __init__(self, name: str):
        """
        Initialize a named registry.

        Args:
            name: Unique name for this registry (e.g., 'models', 'data_loaders')
        """
        self.name = name
        self._registry: Dict[str, Type] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # Register this instance globally
        PluginRegistry._all_registries[name] = self

    def register(
        self,
        key: str,
        override: bool = False,
        **metadata
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a plugin class.

        Args:
            key: Unique identifier for this plugin
            override: If True, allow overriding existing registrations
            **metadata: Additional metadata to store with the registration

        Returns:
            Decorator function

        Raises:
            ValueError: If key already registered and override=False

        Example:
            @model_registry.register("audiomae++", version="2.0")
            class AudioMAEPlusPlus(BaseModel):
                ...
        """
        def decorator(cls: Type[T]) -> Type[T]:
            if key in self._registry and not override:
                raise ValueError(
                    f"Plugin '{key}' already registered in '{self.name}'. "
                    f"Use override=True to replace."
                )

            self._registry[key] = cls
            self._metadata[key] = {
                'class_name': cls.__name__,
                'module': cls.__module__,
                **metadata
            }

            # Store registration info on the class itself
            cls._registry_key = key
            cls._registry_name = self.name

            return cls

        return decorator

    def get(self, key: str) -> Type[T]:
        """
        Get a registered plugin class by key.

        Args:
            key: Plugin identifier

        Returns:
            Plugin class (not instantiated)

        Raises:
            KeyError: If plugin not found
        """
        if key not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Plugin '{key}' not found in '{self.name}' registry. "
                f"Available: {available}"
            )
        return self._registry[key]

    def create(self, key: str, *args, **kwargs) -> T:
        """
        Factory method to instantiate a plugin.

        Args:
            key: Plugin identifier
            *args: Positional arguments for constructor
            **kwargs: Keyword arguments for constructor

        Returns:
            Plugin instance

        Example:
            model = model_registry.create("audiomae++", config)
        """
        cls = self.get(key)
        return cls(*args, **kwargs)

    def list(self) -> List[str]:
        """
        Return list of registered plugin keys.

        Returns:
            List of plugin identifiers
        """
        return list(self._registry.keys())

    def get_metadata(self, key: str) -> Dict[str, Any]:
        """
        Get metadata for a registered plugin.

        Args:
            key: Plugin identifier

        Returns:
            Metadata dict
        """
        if key not in self._metadata:
            raise KeyError(f"Plugin '{key}' not found in '{self.name}'")
        return self._metadata[key]

    def all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all registered plugins.

        Returns:
            Dict mapping keys to metadata
        """
        return dict(self._metadata)

    def __contains__(self, key: str) -> bool:
        """Check if a plugin is registered."""
        return key in self._registry

    def __iter__(self):
        """Iterate over (key, class) pairs."""
        return iter(self._registry.items())

    def __len__(self) -> int:
        """Return number of registered plugins."""
        return len(self._registry)

    def __repr__(self) -> str:
        return f"PluginRegistry('{self.name}', plugins={self.list()})"

    @classmethod
    def get_registry(cls, name: str) -> Optional['PluginRegistry']:
        """
        Get a registry by name.

        Args:
            name: Registry name

        Returns:
            PluginRegistry or None if not found
        """
        return cls._all_registries.get(name)

    @classmethod
    def all_registries(cls) -> Dict[str, 'PluginRegistry']:
        """Return all registered registries."""
        return dict(cls._all_registries)


# Pre-defined registries for common plugin types
model_registry = PluginRegistry("models")
data_loader_registry = PluginRegistry("data_loaders")
transform_registry = PluginRegistry("transforms")
loss_registry = PluginRegistry("losses")


def create_registry(name: str) -> PluginRegistry:
    """
    Create a new plugin registry.

    Args:
        name: Unique name for the registry

    Returns:
        New PluginRegistry instance
    """
    return PluginRegistry(name)
