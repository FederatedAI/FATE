#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import re
import typing
from typing import List, Optional

if typing.TYPE_CHECKING:
    from fate.arch import URI

import pydantic

# see https://www.rfc-editor.org/rfc/rfc3986#appendix-B
# scheme    = $2
# authority = $4
# path      = $5
# query     = $7
# fragment  = $9
_uri_regex = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")


class Metadata(pydantic.BaseModel):
    metadata: dict = pydantic.Field(default_factory=dict)
    name: Optional[str] = None
    namespace: Optional[str] = None
    data_overview: Optional[List] = None

    class Config:
        extra = "forbid"


class ArtifactInputApplySpec(pydantic.BaseModel):
    uri: str
    metadata: Metadata


class ArtifactOutputApplySpec(pydantic.BaseModel):
    uri: str
    _is_template: Optional[bool] = None

    def is_template(self) -> bool:
        return "{index}" in self.uri

    def _check_is_template(self) -> bool:
        return "{index}" in self.uri

    @pydantic.validator("uri")
    def _check_uri(cls, v, values) -> str:
        if not _uri_regex.match(v):
            raise pydantic.ValidationError(f"`{v}` is not valid uri")
        return v
