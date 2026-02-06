from pymilvus import Function
from pydantic_core import core_schema
from typing import Any, Dict, List, Type
from pydantic import GetCoreSchemaHandler

class MilvusFunction(Function):
    @classmethod
    def __get_pydantic_core_schema__(cls, source: Type['MilvusFunction'], _: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        assert source is MilvusFunction
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
    def _validate(value: Dict[str, Any]) -> 'MilvusFunction':
        return MilvusFunction(**value)

    @staticmethod
    def _serialize(value: 'MilvusFunction') -> Dict[str, Any]:
        return value.to_dict()


class MilvusFunctions(List[MilvusFunction]):
    @classmethod
    def __get_pydantic_core_schema__(cls, source: Type['MilvusFunctions'], _: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        assert source is MilvusFunctions
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
    def _validate(value: List[Dict[str, Any]]) -> 'MilvusFunctions':
        instance = MilvusFunctions()
        for item in value:
            instance.add_function(**item)
        return instance

    @staticmethod
    def _serialize(value: 'MilvusFunctions') -> List[MilvusFunction]:
        return list(value)

    def add_function(self, function: MilvusFunction):
        super().append(function)
