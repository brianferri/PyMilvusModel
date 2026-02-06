from enum import Enum
from pydantic import BaseModel
from .function import MilvusFunction, MilvusFunctions
from .index import MilvusIndexParam, MilvusIndexParams
from pymilvus import FieldSchema, CollectionSchema, MilvusClient
from typing import (
    Any,
    Set,
    Dict,
    List,
    Type,
    Union,
    TypeVar,
    ClassVar,
    Optional,
    get_args,
    Annotated,
    get_origin,
)



class MilvusField(FieldSchema):
    def __get_pydantic_json_schema__(self, schema: dict[str, Any]):
        schema.update({
            "title": self.name,
            "type": self.dtype,
            "isPrimary": self.is_primary
        })
        return schema


class MilvusCollectionMetadata(BaseModel):
    """
    Holds some information for registered collection schema

    Attributes:
        cls: the class that defined the collection schema
        indexes: the indexes that the collection uses for lookups
    """
    cls: Type['MilvusModel']
    indexes: MilvusIndexParams = MilvusIndexParams()
    functions: MilvusFunctions = MilvusFunctions()

class MilvusMetadata:
    """
    Interface with the Milvus Model Metadata to initialize the defined collection schemas
    """
    def __init__(self):
        self.collections: Dict[str, MilvusCollectionMetadata] = {}

    def __add_indexes__(self, model: Type['MilvusModel'], indexes: List[MilvusIndexParam]):
        if model.__name__ not in self.collections:
            self.collections[model.__name__] = MilvusCollectionMetadata(cls=model)
        for index in indexes:
            self.collections[model.__name__].indexes.add_index(
                index.field_name,
                index.index_type,
                index.index_name,
                **index._configs
            )

    def __add_functions__(self, model: Type['MilvusModel'], functions: List[MilvusFunction]):
        if model.__name__ not in self.collections:
            self.collections[model.__name__] = MilvusCollectionMetadata(cls=model)
        for function in functions:
            self.collections[model.__name__].functions.add_function(function)
            for output_field in function.output_field_names:
                model._excluded_fields.add(output_field)

    def register(self, model: Type['MilvusModel']):
        if model.__name__ not in self.collections:
            self.collections[model.__name__] = MilvusCollectionMetadata(cls=model)

    def create_all(self, client: MilvusClient):
        self._client = client
        for model_name, metadata in self.collections.items():
            if self._client.has_collection(model_name): continue
            self._client.create_collection(
                model_name,
                schema=metadata.cls._collection_schema,
                index_params=metadata.indexes
            )

T = TypeVar('T', bound="MilvusModel")

class MilvusModel(BaseModel):
    """
    Milvus Model

    The Milvus Model class is a wrapper around the Milvus Client class which changes the programming paradigm
    to be more similiar to SQLModel, allowing the declaration of collections as classes and instantiating
    rows using the same schema.

        .. code-block:: python

            class Model(MilvusModel):
                indexes: list[MilvusIndexParam] = ...

                field: Annotated[
                    # the type shown in the docs/when accessing the property
                    str,
                    # the field passed to the Milvus Vector Store to initialize the collection schema
                    MilvusField(name="field", dtype=DataType.VARCHAR, max_length=256)
                ]
    """
    metadata: ClassVar[MilvusMetadata] = MilvusMetadata()
    _collection_schema: ClassVar[Optional[CollectionSchema]] = None
    _excluded_fields: ClassVar[Set[str]] = set()
    """Fields excluded from insert opertations"""

    auto_id: Optional[bool] = None
    enable_dynamic_field: Optional[bool] = None
    description: Optional[str] = None

    indexes: Optional[List[MilvusIndexParam]] = None
    functions: Optional[List[MilvusFunction]] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls._excluded_fields.update({
            name for name, _ in MilvusModel.__annotations__.items()
        })

        fields = []
        for attr_name, attr_type in cls.__annotations__.items():
            if attr_name == "indexes" and cls.indexes is not None:
                MilvusModel.metadata.__add_indexes__(cls, cls.indexes)
                continue
            if attr_name == "functions" and cls.functions is not None:
                MilvusModel.metadata.__add_functions__(cls, cls.functions)
                continue

            if not get_origin(attr_type) is Annotated:
                raise TypeError("MilvusModel types should be annotated with a MilvusField")

            _, field_schema = get_args(attr_type)
            if not isinstance(field_schema, MilvusField):
                raise TypeError("MilvusModel annotated types must be MilvusFields")
            fields.append(field_schema)

            if field_schema.auto_id:
                cls._excluded_fields.add(attr_name)

        cls._collection_schema = CollectionSchema(
            fields=fields,
            auto_id=getattr(cls, "auto_id", False),
            enable_dynamic_field=getattr(cls, "enable_dynamic_field", False),
            description=getattr(cls, "description", ''),
            functions=getattr(cls, "functions", [])
        )
        MilvusModel.metadata.register(cls)

    @classmethod
    def insert(cls: Type[T], data: Union[T, List[T]], **kwargs):
        """
        Insert data into the collection. Validates data types dynamically.

        Parameters:
            data: An instance or a list of instances of the subclass.
        """
        def validate_record(record: T):
            if not isinstance(record, cls):
                raise TypeError(f"Data must be an instance of {cls.__name__}, got {type(record).__name__}")
            for field_name, field_value in record.__dict__.items():
                field_type = type(field_value)
                if not (isinstance(field_type, type) and issubclass(field_type, Enum)): continue
                if isinstance(field_value, field_type):
                    record.__dict__[field_name] = field_value.value
                elif isinstance(field_value, str):
                    try:
                        record.__dict__[field_name] = field_type(field_value).value
                    except ValueError:
                        raise ValueError(f"Invalid value '{field_value}' for enum field '{field_name}'")
                else:
                    raise TypeError(f"Field '{field_name}' must be of type '{field_type.__name__}' or a valid string.")

        if isinstance(data, cls):
            validate_record(data)
            data_to_insert = [data.model_dump(exclude=cls._excluded_fields)]
        elif isinstance(data, list):
            for record in data: validate_record(record)
            data_to_insert = [item.model_dump(exclude=cls._excluded_fields) for item in data]
        else:
            raise TypeError("Data must be an instance or a list of instances of the subclass.")
        cls.metadata._client.insert(cls.__name__, data_to_insert, **kwargs)

    @classmethod
    def delete(
        cls: Type[T],
        ids: Optional[Union[list, str, int]] = None,
        timeout: Optional[float] = None,
        filter: Optional[str] = "",
        partition_name: Optional[str] = "",
        **kwargs
    ):
        """
        Delete rows

        https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Vector/delete.md
        """
        return cls.metadata._client.delete(cls.__name__, ids, timeout, filter, partition_name, **kwargs)

    @classmethod
    def get(cls: Type[T], ids: List[Any], output_fields: Optional[List[str]], **kwargs) -> List[T]:
        """
        Get entities by ID

        https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Vector/get.md
        """
        return [cls(**vector) for vector in cls.metadata._client.get(cls.__name__, ids, output_fields, **kwargs)]

    @classmethod
    def query(cls: Type[T], filter: str, **kwargs) -> List[T]:
        """
        Scalar filtering with a specified boolean expression

        https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Vector/query.md
        """
        return [cls(**vector) for vector in cls.metadata._client.query(cls.__name__, filter, **kwargs)]
