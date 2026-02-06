from typing import List, Optional
from typing_extensions import Annotated
from pymilvus.orm.constants import CALC_DIST_BM25
from pymilvus import FunctionType, MilvusClient, DataType
from pymilvusmodel import MilvusField, MilvusModel, MilvusIndexParam
from pymilvusmodel.function import MilvusFunction


class ExampleModel(MilvusModel):
    indexes: List[MilvusIndexParam] | None = [
        MilvusIndexParam(
            field_name="vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="collections_vector_index",
            metric_type=CALC_DIST_BM25
        )
    ]
    # ? https://milvus.io/docs/manage-collections.md#Function
    # ? https://milvus.io/docs/full-text-search.md#Define-the-BM25-function
    functions: List[MilvusFunction] | None = [
        MilvusFunction(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["vector"],
            function_type=FunctionType.BM25,
        )
    ]

    id: Annotated[
        Optional[int],
        MilvusField(name="id", dtype=DataType.INT64,
                    is_primary=True, auto_id=True)
    ] = None
    text: Annotated[
        str,
        MilvusField(name="text", dtype=DataType.VARCHAR, max_length=256, enable_analyzer=True)
    ]
    # ! The vector field gets populated automatically from the text field using the BM25 function
    vector: Annotated[
        List[float],
        MilvusField(name="vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
    ] = []


MILVUS_CLIENT = MilvusClient("http://localhost:19530")
MilvusModel.metadata.create_all(MILVUS_CLIENT)
print(MILVUS_CLIENT.list_collections()) # ['ExampleModel']
ExampleModel.insert(ExampleModel(text="Hello, World!"))
print(ExampleModel.query(filter="id>=0"))
"""
[
    ...
    ExampleModel(
        auto_id=None,
        enable_dynamic_field=None,
        description=None,
        indexes=[{
            'field_name': 'vector',
            'index_type': 'SPARSE_INVERTED_INDEX',
            'index_name': 'collections_vector_index',
            'metric_type': 'BM25'
        }],
        functions=[{
            'name': 'text_bm25_emb',
            'description': '',
            'type': <FunctionType.BM25: 1>,
            'input_field_names': ['text'],
            'output_field_names': ['vector'],
            'params': {}
        }],
        id=464090798726323824,
        text='Hello, World!',
        vector=[]
    )
    ...
]
"""

