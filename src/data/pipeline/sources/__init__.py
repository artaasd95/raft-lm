"""Data source connectors for the RAFT-LM data platform."""

from typing import Dict, Type

from src.data.pipeline.sources.base import BaseSource
from src.data.pipeline.sources.databricks import DatabricksSource
from src.data.pipeline.sources.file_source import FileSource
from src.data.pipeline.sources.hf_source import HuggingFaceSource
from src.data.pipeline.sources.sql_source import SQLSource

SOURCE_TYPES: Dict[str, Type[BaseSource]] = {
    "file": FileSource,
    "hf": HuggingFaceSource,
    "sql": SQLSource,
    "databricks": DatabricksSource,
}


def build_source(spec: dict) -> BaseSource:
    source_type = spec.get("type")
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"Unsupported source type: {source_type}")
    return SOURCE_TYPES[source_type].from_spec(spec)


__all__ = [
    "BaseSource",
    "FileSource",
    "HuggingFaceSource",
    "SQLSource",
    "DatabricksSource",
    "build_source",
    "SOURCE_TYPES",
]
