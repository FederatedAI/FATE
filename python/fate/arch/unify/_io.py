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
import hashlib
import re
from abc import ABCMeta
from dataclasses import dataclass
from typing import Optional

# see https://www.rfc-editor.org/rfc/rfc3986#appendix-B
# scheme    = $2
# authority = $4
# path      = $5
# query     = $7
# fragment  = $9
_uri_regex = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")


@dataclass
class URI:
    schema: str
    path: str
    query: Optional[str] = None
    fragment: Optional[str] = None
    authority: Optional[str] = None

    @classmethod
    def from_string(cls, uri: str) -> "URI":
        match = _uri_regex.fullmatch(uri)
        if match is None:
            raise ValueError(f"`{uri}` is not valid uri")
        _, schema, _, authority, path, _, query, _, fragment = match.groups()
        return URI(schema, path, query, fragment, authority)

    def to_schema(self):
        for cls in ConcrateURI.__subclasses__():
            if cls.schema() == self.schema:
                return cls.from_uri(self)
        raise NotImplementedError(f"uri schema `{self.schema}` not found")


class ConcrateURI(metaclass=ABCMeta):
    @classmethod
    def schema(cls) -> str:
        ...

    @classmethod
    def from_uri(cls, uri: URI) -> "ConcrateURI":
        ...

    def create_file(self, name):
        ...

    def to_string(self):
        ...


_EGGROLL_NAME_MAX_SIZE = 128


@dataclass
class FileURI(ConcrateURI):
    path: str

    @classmethod
    def schema(cls):
        return "file"

    @classmethod
    def from_uri(cls, uri: URI):
        return FileURI(uri.path)

    def create_file(self, name):
        return FileURI(f"{self.path}/{name}")

    def to_string(self):
        return f"file://{self.path}"


@dataclass
class EggrollURI(ConcrateURI):
    namespace: str
    name: str

    @classmethod
    def schema(cls):
        return "eggroll"

    @classmethod
    def from_uri(cls, uri: URI):
        _, namespace, *names = uri.path.split("/")
        name = "_".join(names)
        if len(name) > _EGGROLL_NAME_MAX_SIZE:
            name = hashlib.md5(name.encode(encoding="utf8")).hexdigest()[:_EGGROLL_NAME_MAX_SIZE]
        return EggrollURI(namespace, name)

    def create_file(self, name):
        name = f"{self.name}_{name}"
        if len(name) > _EGGROLL_NAME_MAX_SIZE:
            name = hashlib.md5(name.encode(encoding="utf8")).hexdigest()[:_EGGROLL_NAME_MAX_SIZE]
        return EggrollURI(namespace=self.namespace, name=name)

    def to_string(self):
        return f"eggroll:///{self.namespace}/{self.name}"


@dataclass
class HdfsURI(ConcrateURI):
    path: str
    authority: Optional[str] = None

    @classmethod
    def schema(cls):
        return "hdfs"

    @classmethod
    def from_uri(cls, uri: URI):
        return HdfsURI(uri.path, uri.authority)

    def create_file(self, name):
        return HdfsURI(path=f"{self.path}/{name}", authority=self.authority)

    def to_string(self):
        if self.authority:
            return f"hdfs://{self.authority}{self.path}"
        else:
            return f"hdfs://{self.path}"


@dataclass
class HttpURI(ConcrateURI):
    path: str
    authority: Optional[str] = None

    @classmethod
    def schema(cls):
        return "http"

    @classmethod
    def from_uri(cls, uri: URI):
        return HttpURI(uri.path, uri.authority)

    def create_file(self, name):
        return HttpURI(path=f"{self.path}/{name}", authority=self.authority)

    def to_string(self):
        if self.authority:
            return f"http://{self.authority}{self.path}"
        else:
            return f"http://{self.path}"


@dataclass
class HttpsURI(ConcrateURI):
    path: str
    authority: Optional[str] = None

    @classmethod
    def schema(cls):
        return "https"

    @classmethod
    def from_uri(cls, uri: URI):
        return HttpsURI(uri.path, uri.authority)

    def create_file(self, name):
        return HttpURI(path=f"{self.path}/{name}", authority=self.authority)

    def to_string(self):
        if self.authority:
            return f"https://{self.authority}{self.path}"
        else:
            return f"https://{self.path}"
