"""
Utility subpackage for HIIT_maker.

This module exposes frequently used helpers for JSON parsing,
logging, and global state management.
"""

from .state import _CHOICE_STATE
from .state_utils import reset_choice_state
from .json_tools import _coerce_json
from .logging_utils import _log

__all__ = [
    "_CHOICE_STATE",
    "reset_choice_state",
    "_coerce_json",
    "_log",
]
