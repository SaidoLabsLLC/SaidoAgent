"""Sample Python file for ingest pipeline tests."""

import os
from pathlib import Path


def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Calculator:
    """A simple calculator."""

    def __init__(self, value: float = 0):
        self._value = value

    def add(self, n: float) -> "Calculator":
        self._value += n
        return self

    def result(self) -> float:
        return self._value


class AdvancedCalculator(Calculator):
    """Calculator with extra operations."""

    def multiply(self, n: float) -> "AdvancedCalculator":
        self._value *= n
        return self
