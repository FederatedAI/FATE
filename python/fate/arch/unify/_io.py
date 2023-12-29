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
from typing import List, Optional

# see https://www.rfc-editor.org/rfc/rfc3986#appendix-B
# scheme    = $2
# authority = $4
# path      = $5
# query     = $7
# fragment  = $9
_uri_regex = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")


class URI:
    def __init__(
        self,
        scheme: str,
        path: str,
        query: Optional[str] = None,
        fragment: Optional[str] = None,
        authority: Optional[str] = None,
        original_uri: Optional[str] = None,
    ):
        self.scheme = scheme
        self.path = path
        self.query = query
        self.fragment = fragment
        self.authority = authority

        self.original_uri = original_uri
        if self.original_uri is None:
            self.original_uri = self.to_string()

    @classmethod
    def from_string(cls, uri: str) -> "URI":
        match = _uri_regex.fullmatch(uri)
        if match is None:
            raise ValueError(f"`{uri}` is not valid uri")
        _, scheme, _, authority, path, _, query, _, fragment = match.groups()
        return URI(scheme=scheme, path=path, query=query, fragment=fragment, authority=authority, original_uri=uri)

    def to_string(self) -> str:
        uri = ""
        if self.scheme:
            uri += f"{self.scheme}:"
        if self.authority:
            uri += f"//{self.authority}"
        elif self.scheme:
            uri += f"//"
        uri += self.path
        if self.query:
            uri += f"?{self.query}"
        if self.fragment:
            uri += f"#{self.fragment}"
        return uri

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def path_splits(self) -> List[str]:
        parts = self.path.split("/")
        return parts
