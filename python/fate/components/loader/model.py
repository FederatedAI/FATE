import json
import tarfile
import tempfile
from datetime import datetime
from typing import List

import pydantic
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
        super().__init__(tarfile.open(uri.path, "w"))

    def close(self):
        self.tar.close()

    def mlmd_send(self, mlmd, artifact, metadata):
        import requests

        # TODO: upload
        response = requests.post(url=self.uri.to_string(), json={"data": self.memory_file})

        mlmd.log_output_model(artifact.name, artifact, metadata=metadata, tarfile=self)


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

    def write_model(self, name, model, metadata, created_time=None):
        if created_time is None:
            created_time = datetime.now()
        self._get_tar().add_model(name, model)
        self.models.append(MLModelModelSpec(name=name, created_time=created_time, metadata=metadata))


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
        import requests

        # TODO: download tar
        requests.get(url=uri.to_string()).json().get("data")
        tar = ...
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


class MLModelComponentSpec(pydantic.BaseModel):
    name: str
    provider: str
    version: str
    metadata: dict


class MLModelPartiesSpec(pydantic.BaseModel):
    guest: List[str]
    host: List[str]
    arbiter: List[str]


class MLModelFederatedSpec(pydantic.BaseModel):

    task_id: str
    parties: MLModelPartiesSpec
    component: MLModelComponentSpec


class MLModelModelSpec(pydantic.BaseModel):
    name: str
    created_time: datetime
    metadata: dict


class MLModelPartySpec(pydantic.BaseModel):

    party_task_id: str
    role: str
    partyid: str
    models: List[MLModelModelSpec]


class MLModelSpec(pydantic.BaseModel):

    federated: MLModelFederatedSpec
    party: MLModelPartySpec
