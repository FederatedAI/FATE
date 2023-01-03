import logging

import requests

from ....unify import URI

logger = logging.getLogger(__name__)


class HTTPModelWriter:
    def __init__(self, ctx, name: str, uri: URI, metadata) -> None:
        self.name = name
        self.ctx = ctx
        self.entrypoint = f"{self.uri.schema}://{self.uri.authority}{self.uri.path}"
        self.uri = uri

    def write_model(self, model):
        logger.debug(self.entrypoint)
        response = requests.post(url=self.entrypoint, json={"data": model})
        logger.debug(response.text)


class HTTPModelReader:
    def __init__(self, ctx, name: str, uri: URI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.entrypoint = f"{self.uri.schema}://{self.uri.authority}{self.uri.path}"
        self.metadata = metadata

    def read_model(self):
        return requests.get(url=self.entrypoint).json().get("data", {})
