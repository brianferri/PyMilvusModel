from enum import Enum
from typing import List, Optional, Set
from numpy import dtype
from typing_extensions import Annotated
from pymilvus import MilvusClient, DataType
from pymilvusmodel import MilvusField, MilvusModel
from pymilvusmodel.index import MilvusIndexParam

class ExampleEnum(Enum):
    FIRST = "One"
    SECOND = "Two"

class ExampleModel(MilvusModel):
    indexes: list[MilvusIndexParam] | None = [
        MilvusIndexParam("vector", "IVF_FLAT", "vector_index", metric_type="COSINE", params={
            "nlist": 128
        })
    ]

    id: Annotated[
        Optional[int],
        MilvusField(name="id", dtype=DataType.INT64,
                    is_primary=True, auto_id=True)
    ] = None
    enum: Annotated[
        ExampleEnum,
        MilvusField(name="enum", dtype=DataType.VARCHAR, max_length=256)
    ]
    enum_list: Annotated[
        Set[ExampleEnum],
        MilvusField(name="enum_list",
                    dtype=DataType.ARRAY, element_type=DataType.VARCHAR,
                    max_capacity=128, max_length=256)
    ]
    vector: Annotated[
        List[float],
        MilvusField(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2)
    ]

MILVUS_CLIENT = MilvusClient("http://localhost:19530")
MilvusModel.metadata.create_all(MILVUS_CLIENT)
print(MILVUS_CLIENT.list_collections()) # ['ExampleModel']
ExampleModel.insert(ExampleModel(
    enum=ExampleEnum.FIRST,
    enum_list={
        ExampleEnum.FIRST,
        ExampleEnum.SECOND,
    },
    vector=[0, 1]
))
print(ExampleModel.query(filter="id>=0"))

