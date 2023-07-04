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
import json
import logging
import tarfile
import tempfile
from datetime import datetime

from fate.components.core.spec.model import (
    MLModelComponentSpec,
    MLModelFederatedSpec,
    MLModelModelSpec,
    MLModelPartiesSpec,
    MLModelPartySpec,
    MLModelSpec,
)
from ruamel import yaml


def load_output_model_wrapper(task_id, party_task_id, cpn, role, partyid, federation):
    return ComponentModelWriterWrapper(cpn, federation, task_id, party_task_id, role, partyid)


def load_input_model_wrapper():
    return ComponentModelLoaderWrapper()


_MODEL_META_NAME = "FMLModel.yaml"


class ComponentModelWriterWrapper:
    def __init__(self, cpn, federation, task_id, party_task_id, role, party_id) -> None:
        self.task_id = task_id
        self.party_task_id = party_task_id
        self.role = role
        self.party_id = party_id
        self.cpn_spec = MLModelComponentSpec(name=cpn.name, provider=cpn.provider, version=cpn.version, metadata={})
        guest = []
        host = []
        arbiter = []
        for party in federation.metadata.parties.parties:
            if party.role == "guest":
                guest.append(party.partyid)
            if party.role == "host":
                host.append(party.partyid)
            if party.role == "arbiter":
                arbiter.append(party.partyid)
        self.parties_spec = MLModelPartiesSpec(guest=guest, host=host, arbiter=arbiter)

    def wrap(self, artifact, io_mlmd):
        return ComponentModelWriter(self, artifact, io_mlmd)


class ComponentModelLoaderWrapper:
    def wrap(self, artifact, io_mlmd):
        return ComponentModelLoader(artifact, io_mlmd)


class ModelTarWriteHandler:
    def __init__(self, tar) -> None:
        self.tar = tar

    def add_model(self, name, model, file_format):
        if file_format == "json":
            self.add_json_model(name, model)
        else:
            raise NotImplementedError(f"file_format={file_format} not support")

    def add_json_model(self, name, model):
        with tempfile.NamedTemporaryFile("w") as f:
            json.dump(model, f)
            f.flush()
            self.tar.add(f.name, name)

    def add_meta(self, meta):
        with tempfile.NamedTemporaryFile("w") as f:
            yaml.safe_dump(meta, f)
            f.flush()
            self.tar.add(f.name, _MODEL_META_NAME)


class FileModelTarWriteHandler(ModelTarWriteHandler):
    def __init__(self, uri) -> None:
        super().__init__(tarfile.open(uri.path, "w"))

    def close(self):
        self.tar.close()

    def mlmd_send(self, mlmd, artifact, metadata):
        mlmd.log_output_model(artifact.name, artifact, metadata=metadata)


class HttpModelTarWriteTarHandler(ModelTarWriteHandler):
    def __init__(self, uri) -> None:
        self.uri = uri
        import io

        self.memory_file = io.BytesIO()
        super().__init__(tarfile.open(fileobj=self.memory_file, mode="w"))

    def close(self):
        self.tar.close()

    def mlmd_send(self, mlmd, artifact, metadata):
        import requests

        logging.info(f"mlmd send uri: {self.uri.to_string()}")
        self.memory_file.seek(0)
        response = requests.post(url=self.uri.to_string(), files={"file": self.memory_file})
        logging.info(f"response: {response.text}")
        mlmd.log_output_model(artifact.name, artifact, metadata=metadata)


class ComponentModelWriter:
    def __init__(self, info: ComponentModelWriterWrapper, artifact, mlmd) -> None:
        self.info = info
        self.models = []

        from fate.arch.unify import URI

        self.artifact = artifact
        self.uri = URI.from_string(artifact.uri).to_schema()
        self.mlmd = mlmd

        self._tar = None

    def __enter__(self):
        from fate.arch.unify import FileURI, HttpsURI, HttpURI

        if isinstance(self.uri, FileURI):
            self._tar = FileModelTarWriteHandler(self.uri)
        elif isinstance(self.uri, (HttpURI, HttpsURI)):
            self._tar = HttpModelTarWriteTarHandler(self.uri)
        else:
            raise NotImplementedError(f"model writer not support uri: {self.uri}")
        return self

    def __exit__(self, type, value, trace):
        self._write_meta()
        self._get_tar().mlmd_send(self.mlmd, self.artifact, self._get_meta().dict())
        self._get_tar().close()

    def _get_tar(self):
        if self._tar is None:
            raise ValueError(f"should open first")
        return self._tar

    def _get_meta(self):
        return MLModelSpec(
            federated=MLModelFederatedSpec(
                task_id=self.info.task_id, parties=self.info.parties_spec, component=self.info.cpn_spec
            ),
            party=MLModelPartySpec(
                party_task_id=self.info.party_task_id,
                role=self.info.role,
                partyid=self.info.party_id,
                models=self.models,
            ),
        )

    def _write_meta(self):
        self._get_tar().add_meta(self._get_meta().dict())

    def write_model(self, name, model, metadata, file_format="json", created_time=None):
        if created_time is None:
            created_time = datetime.now()
        self._get_tar().add_model(name, model, file_format=file_format)
        self.models.append(
            MLModelModelSpec(name=name, created_time=created_time, file_format=file_format, metadata=metadata)
        )


class ModelTarReadHandler:
    def __init__(self, tar) -> None:
        self.tar = tar
        self.meta = None

    def add_model(self, name, model):
        with tempfile.NamedTemporaryFile("w") as f:
            json.dump(model, f)
            f.flush()
            self.tar.add(f.name, name)

    def add_meta(self, meta):
        with tempfile.NamedTemporaryFile("w") as f:
            yaml.safe_dump(meta, f)
            f.flush()
            self.tar.add(f.name, _MODEL_META_NAME)

    def get_meta(self):
        if self.meta is None:
            with tempfile.TemporaryDirectory() as d:
                path = f"{d}/{_MODEL_META_NAME}"
                self.tar.extract(_MODEL_META_NAME, d)
                with open(path, "r") as f:
                    meta = yaml.safe_load(f)

            self.meta = MLModelSpec.parse_obj(meta)
        return self.meta

    def read_model(self, **kwargs):
        # return first for now, TODO: extend this
        model_info = self.get_meta().party.models[0]
        model_name = model_info.name
        file_format = model_info.file_format
        if file_format == "json":
            return self.read_json_model(model_name)
        else:
            raise NotImplementedError(f"file_format={file_format} not supported")

    def read_json_model(self, model_name):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/{model_name}"
            self.tar.extract(model_name, d)
            with open(path, "r") as f:
                return json.load(f)


class FileModelTarReadHandler(ModelTarReadHandler):
    def __init__(self, uri) -> None:
        super().__init__(tarfile.open(uri.path, "r"))

    def close(self):
        self.tar.close()


class HttpModelTarReadTarHandler(ModelTarReadHandler):
    def __init__(self, uri) -> None:
        import io
        from contextlib import closing

        import requests

        memory_file = io.BytesIO()
        logging.debug(f"read model from: {uri.to_string()}")
        with closing(requests.get(url=uri.to_string(), stream=True)) as response:
            for chunk in response.iter_content(1024):
                if chunk:
                    memory_file.write(chunk)
        memory_file.seek(0)
        tar = tarfile.open(fileobj=memory_file, mode="r")
        logging.debug(f"read model success")
        super().__init__(tar)

    def close(self):
        self.tar.close()


class ComponentModelLoader:
    def __init__(self, artifact, mlmd) -> None:
        self.artifact = artifact
        from fate.arch.unify import URI

        self.uri = URI.from_string(artifact.uri).to_schema()
        self.mlmd = mlmd
        self._tar = None
        self._meta = None

    def __enter__(self):
        from fate.arch.unify import FileURI, HttpsURI, HttpURI

        if isinstance(self.uri, FileURI):
            self._tar = FileModelTarReadHandler(self.uri)
        elif isinstance(self.uri, (HttpURI, HttpsURI)):
            self._tar = HttpModelTarReadTarHandler(self.uri)
        else:
            raise NotImplementedError(f"model writer not support uri: {self.uri}")
        return self

    def __exit__(self, type, value, trace):
        self._get_tar().close()

    def _get_tar(self):
        if self._tar is None:
            raise ValueError(f"should open first")
        return self._tar

    def read_model(self, **kwargs):
        return self._get_tar().read_model(**kwargs)
