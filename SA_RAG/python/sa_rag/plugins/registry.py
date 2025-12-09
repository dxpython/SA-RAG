"""Plugin Registry
Manages plugins for SA-RAG
"""

from typing import Dict, List, Optional
import rust_core


class PluginRegistry:
    """Registry for managing plugins"""
    
    def __init__(self, engine: Optional[rust_core.RustCoreEngine] = None):
        """Initialize plugin registry
        
        Args:
            engine: Rust core engine instance
        """
        self.engine = engine
        self.registry = rust_core.PluginRegistry() if engine else None
        self._python_plugins: Dict[str, object] = {}
    
    def register_plugin(
        self,
        plugin_id: str,
        plugin_type: str,
        name: str,
        plugin: object,
    ):
        """Register a Python plugin
        
        Args:
            plugin_id: Unique plugin identifier
            plugin_type: Type of plugin ("ranker", "parser", "graph_policy")
            name: Plugin name
            plugin: Plugin instance
        """
        self._python_plugins[plugin_id] = plugin
        
        if self.registry:
            metadata = rust_core.PluginMetadata(
                plugin_id=plugin_id,
                plugin_type=plugin_type,
                name=name,
            )
            self.registry.register_metadata(metadata)
    
    def get_plugin(self, plugin_id: str) -> Optional[object]:
        """Get plugin by ID
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Plugin instance or None
        """
        return self._python_plugins.get(plugin_id)
    
    def list_plugins(self) -> List[Dict]:
        """List all registered plugins
        
        Returns:
            List of plugin metadata dictionaries
        """
        if self.registry:
            plugins = self.registry.list_plugins()
            return [
                {
                    "plugin_id": p.plugin_id,
                    "plugin_type": p.plugin_type,
                    "name": p.name,
                    "version": p.version,
                    "author": p.author,
                    "description": p.description,
                    "enabled": p.enabled,
                }
                for p in plugins
            ]
        else:
            return []

