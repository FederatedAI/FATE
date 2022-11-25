from typing import Literal

import pydantic


class StandaloneComputingSpec(pydantic.BaseModel):
    class MetadataSpec(pydantic.BaseModel):
        computing_id: str

    type: Literal["standalone"]
    metadata: MetadataSpec


class EggrollComputingSpec(pydantic.BaseModel):
    class MetadataSpec(pydantic.BaseModel):
        computing_id: str

    type: Literal["eggroll"]
    metadata: MetadataSpec


class SparkComputingSpec(pydantic.BaseModel):
    class MetadataSpec(pydantic.BaseModel):
        computing_id: str

    type: Literal["spark"]
    metadata: MetadataSpec


class CustomComputingSpec(pydantic.BaseModel):
    type: Literal["custom"]
    metadata: dict
