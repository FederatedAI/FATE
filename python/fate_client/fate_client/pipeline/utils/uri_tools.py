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
"""
this file aims to process uri
"""

import urllib
from urllib.parse import urlparse
from ..conf.types import UriTypes


def get_schema_from_uri(uri: str):
    source = parse_uri(uri)
    if not source.scheme or source.scheme == "file":
        return UriTypes.LOCAL

    if source.scheme == "lmdb":
        return UriTypes.LMDB

    if source.scheme == "sql":
        return UriTypes.SQL

    return source.scheme


def parse_uri(uri: str) -> 'urllib.parse.ParseResult':
    return urlparse(uri)


def replace_uri_path(uri: urllib.parse.ParseResult, path) -> 'urllib.parse.ParseResult':
    return uri._replace(path=path)
