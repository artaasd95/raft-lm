"""Training-time generation stubs."""

from src.generation.base import BaseGenerator
from src.generation.mock import MockGenerator, build_generator

__all__ = ["BaseGenerator", "MockGenerator", "build_generator"]
