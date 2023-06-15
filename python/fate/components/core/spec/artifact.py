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
from typing import Optional

import pydantic

# see https://www.rfc-editor.org/rfc/rfc3986#appendix-B
# scheme    = $2
# authority = $4
# path      = $5
# query     = $7
# fragment  = $9
_uri_regex = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")


class ArtifactInputApplySpec(pydantic.BaseModel):
    uri: str
    metadata: dict = pydantic.Field(default_factory=dict)

    def get_uri(self) -> "URI":
        return URI.from_string(self.uri)


class ArtifactOutputApplySpec(pydantic.BaseModel):
    uri: str
    _is_template: Optional[bool] = None

    def get_uri(self, index) -> "URI":
        if self.is_template():
            return URI.from_string(self.uri.format(index=index))
        else:
            if index != 0:
                raise ValueError(f"index should be 0, but got {index}")
            return URI.from_string(self.uri)

    def is_template(self) -> bool:
        return "{index}" in self.uri

    def _check_is_template(self) -> bool:
        return "{index}" in self.uri

    @pydantic.validator("uri")
    def _check_uri(cls, v, values) -> str:
        if not _uri_regex.match(v):
            raise pydantic.ValidationError(f"`{v}` is not valid uri")
        return v


class URI:
    def __init__(
        self,
        schema: str,
        path: str,
        query: Optional[str] = None,
        fragment: Optional[str] = None,
        authority: Optional[str] = None,
    ):
        self.schema = schema
        self.path = path
        self.query = query
        self.fragment = fragment
        self.authority = authority

    @classmethod
    def from_string(cls, uri: str) -> "URI":
        match = _uri_regex.fullmatch(uri)
        if match is None:
            raise ValueError(f"`{uri}` is not valid uri")
        _, schema, _, authority, path, _, query, _, fragment = match.groups()
        return URI(schema=schema, path=path, query=query, fragment=fragment, authority=authority)
