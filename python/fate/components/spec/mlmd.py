from typing import Literal

from pydantic import BaseModel


class PipelineMLMDSpec(BaseModel):
    class PipelineMLMDMetaData(BaseModel):
        db: str

    type: Literal["pipeline"]
    metadata: PipelineMLMDMetaData


class FlowMLMDSpec(BaseModel):
    class FlowMLMDMetaData(BaseModel):
        entrypoint: str

    type: Literal["flow"]
    metadata: FlowMLMDMetaData


class CustomMLMDSpec(BaseModel):
    class CustomMLMDMetaData(BaseModel):
        entrypoint: str

    type: Literal["custom"]
    name: str
    metadata: CustomMLMDMetaData
