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
