from typing import Any, Dict, List, Type
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from pymilvus.milvus_client import IndexParams
from pymilvus.milvus_client.index import IndexParam

class MilvusIndexParam(IndexParam):
    @classmethod
    def __get_pydantic_core_schema__(cls, source: Type['MilvusIndexParam'], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        assert source is MilvusIndexParam
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.dict_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize,
                info_arg=False,
                return_schema=core_schema.dict_schema(),
            ),
        )

    @staticmethod
    def _validate(value: Dict[str, Any]) -> 'MilvusIndexParam':
        return MilvusIndexParam(**value)

    @staticmethod
    def _serialize(value: 'MilvusIndexParam') -> Dict[str, Any]:
        return dict(value)


class MilvusIndexParams(IndexParams):
    @classmethod
    def __get_pydantic_core_schema__(cls, source: Type['MilvusIndexParams'], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        assert source is MilvusIndexParams
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.list_schema(core_schema.dict_schema()),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize,
                info_arg=False,
                return_schema=core_schema.list_schema(core_schema.dict_schema()),
            ),
        )

    @staticmethod
    def _validate(value: List[Dict[str, Any]]) -> 'MilvusIndexParams':
        instance = MilvusIndexParams()
        for item in value:
            instance.add_index(**item)
        return instance

    @staticmethod
    def _serialize(value: 'MilvusIndexParams') -> List[Dict[str, Any]]:
        return list(value)
