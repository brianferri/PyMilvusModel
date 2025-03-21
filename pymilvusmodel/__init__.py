"""
PyMilvusModel

Inspired by [SQLModel](https://sqlmodel.tiangolo.com/), PyMilvusModel:

> is a library for interacting with [Milvus](https://milvus.io/) databases from Python code, with Python objects.

and follows the same paradigm, using Python type annotations and powered by [Pydantic](https://docs.pydantic.dev/latest/)
"""

__version__ = "0.1.1"
__author__ = 'Brian Ferri'

from .model import (
    MilvusModel,
    MilvusField
)
from .index import (
    MilvusIndexParam,
    MilvusIndexParams
)

__all__ = [
    "MilvusModel",
    "MilvusField",
    "MilvusIndexParam",
    "MilvusIndexParams",
]
