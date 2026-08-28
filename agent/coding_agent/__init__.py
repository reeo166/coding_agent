"""A framework-free local coding agent."""

from .agent import CodingAgent
from .api import ChatCompletionClient
from .config import Settings
from .tools import ToolRegistry

__all__ = ["ChatCompletionClient", "CodingAgent", "Settings", "ToolRegistry"]
__version__ = "0.1.0"
