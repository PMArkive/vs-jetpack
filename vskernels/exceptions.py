from __future__ import annotations

from typing import Any

from vstools import CustomValueError, FuncExceptT

__all__ = [
    'UnknownKernelError',
]


class UnknownKernelError(CustomValueError):
    """Raised when an unknown kernel is passed."""

    def __init__(
        self, function: FuncExceptT, kernel: str, message: str = 'Unknown kernel: {kernel}!',
        **kwargs: Any
    ) -> None:
        super().__init__(message, function, kernel=kernel, **kwargs)
