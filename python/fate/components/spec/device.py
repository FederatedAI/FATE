from typing import Literal

import pydantic


class CPUSpec(pydantic.BaseModel):
    type: Literal["CPU"]
    metadata: dict = {}


class GPUSpec(pydantic.BaseModel):
    type: Literal["GPU"]
    metadata: dict = {}
