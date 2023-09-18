from typing import Literal, Union

import pydantic


class PaillierCipherParam(pydantic.BaseModel):
    method: Literal["paillier"] = "paillier"
    key_length: pydantic.conint(gt=1024) = 1024


class NoopCipher(pydantic.BaseModel):
    method: Literal[None]


CipherParamType = Union[PaillierCipherParam, NoopCipher]
