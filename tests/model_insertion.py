from typing import List, Optional
from typing_extensions import Annotated
from pymilvus import MilvusClient, DataType
from pymilvusmodel import MilvusIndexParam, MilvusField, MilvusModel


class ExampleModel(MilvusModel):
    indexes: list[MilvusIndexParam] = [
        MilvusIndexParam("vector", "IVF_FLAT", "vector_index", metric_type="COSINE", params={
            "nlist": 128
        })
    ]
    id: Annotated[
        Optional[int],
        MilvusField(name="id", dtype=DataType.INT64,
                    is_primary=True, auto_id=True)
    ] = None
    vector: Annotated[
        List[float],
        MilvusField(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2)
    ]


MILVUS_CLIENT = MilvusClient("test.db")
MilvusModel.metadata.create_all(MILVUS_CLIENT)

ExampleModel.insert(ExampleModel(
    vector=[0, 1]
))
