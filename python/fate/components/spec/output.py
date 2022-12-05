from typing import Literal, Union

import pydantic


class DirectoryDataPool(pydantic.BaseModel):
    class DirectoryDataPoolMetadata(pydantic.BaseModel):
        uri: str
        format: str = "csv"

    type: Literal["directory"]
    metadata: DirectoryDataPoolMetadata


class CustomDataPool(pydantic.BaseModel):
    type: Literal["custom"]
    metadata: dict


class DirectoryModelPool(pydantic.BaseModel):
    class DirectoryDataPoolMetadata(pydantic.BaseModel):
        uri: str
        format: str = "json"

    type: Literal["directory"]
    metadata: DirectoryDataPoolMetadata


class CustomModelPool(pydantic.BaseModel):
    type: Literal["custom"]
    metadata: dict


class DirectoryMetricPool(pydantic.BaseModel):
    class DirectoryDataPoolMetadata(pydantic.BaseModel):
        uri: str
        format: str = "json"

    type: Literal["directory"]
    metadata: DirectoryDataPoolMetadata


class CustomMetricPool(pydantic.BaseModel):
    type: Literal["custom"]
    metadata: dict


class OutputPoolConf(pydantic.BaseModel):
    data: Union[DirectoryDataPool, CustomDataPool]
    model: Union[DirectoryModelPool, CustomModelPool]
    metric: Union[DirectoryMetricPool, CustomMetricPool]
