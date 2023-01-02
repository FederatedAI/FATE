import json
import tarfile
import tempfile
from datetime import datetime
from typing import List

import pydantic
from ruamel import yaml


def load_output_model_wrapper(jobid, taskid, cpn, role, partyid, federation):
    return ComponentModelWriterWrapper(cpn, federation, jobid, taskid, role, partyid)


def load_input_model_wrapper():
    return ComponentModelLoaderWrapper()


class ComponentModelWriterWrapper:
    def __init__(self, cpn, federation, jobid, taskid, role, partyid) -> None:
        self.jobid = jobid
        self.taskid = taskid
        self.role = role
        self.partyid = partyid
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


class ComponentModelWriter:
    def __init__(self, info: ComponentModelWriterWrapper, artifact, mlmd) -> None:
        self.info = info
        self.models = []

        from fate.arch.unify import URI

        self.artifact = artifact
        self.uri = URI.from_string(artifact.uri)
        self.mlmd = mlmd
        self._tar = None

    def __enter__(self):
        self._tar = tarfile.open(self.uri.path, "w")
        return self

    def __exit__(self, type, value, trace):
        if self._tar is None:
            raise ValueError(f"should open first")
        self._write_meta()
        self._mlmd_send()
        self._tar.close()

    def _mlmd_send(self):
        metadata = self._get_meta().json()
        self.mlmd.log_output_model(self.artifact.name, self.artifact, metadata)

    def _add(self, path, name):
        if self._tar is None:
            raise ValueError(f"should open first")
        self._tar.add(path, name)

    def _get_meta(self):
        return MLModelSpec(
            federated=MLModelFederatedSpec(
                jobid=self.info.jobid, parties=self.info.parties_spec, component=self.info.cpn_spec
            ),
            party=MLModelPartySpec(
                party_task_id=self.info.taskid, role=self.info.role, partyid=self.info.partyid, models=self.models
            ),
        )

    def _write_meta(self):
        with tempfile.NamedTemporaryFile("w") as f:
            yaml.safe_dump(self._get_meta().json(), f)
            f.flush()
            self._add(f.name, "FMLModel.yaml")

    def write_model(self, name, model, metadata, created_time=None):
        if created_time is None:
            created_time = datetime.now()
        with tempfile.NamedTemporaryFile("w") as f:
            json.dump(model, f)
            f.flush()
            self._add(f.name, name)
            self.models.append(MLModelModelSpec(name=name, created_time=created_time, metadata=metadata))


class ComponentModelLoader:
    def __init__(self, artifact, mlmd) -> None:
        self.artifact = artifact
        from fate.arch.unify import URI

        self.uri = URI.from_string(artifact.uri)
        self.mlmd = mlmd
        self._tar = None
        self._meta = None

    def __enter__(self):
        self._tar = tarfile.open(self.uri.path, "r")
        return self

    def __exit__(self, type, value, trace):
        if self._tar is None:
            raise ValueError(f"should open first")
        self._tar.close()

    def _get_meta(self):
        if self._meta is None:
            if self._tar is None:
                raise ValueError(f"should open first")
            with tempfile.TemporaryDirectory() as d:
                path = f"{d}/FMLModel.yaml"
                self._tar.extract("FMLModel.yaml", path)
                with open(path, "r") as f:
                    meta = yaml.safe_load(f)

            self._meta = MLModelSpec.parse_obj(meta)
        return self._meta

    def read_model(self, **kwargs):
        if self._tar is None:
            raise ValueError(f"should open first")
        # return first for now, TODO: extend this
        model_info = self._get_meta().party.models[0]
        model_name = model_info.name
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/{model_name}"
            self._tar.extract(model_name, path)
            with open(model_name, "r") as f:
                return json.load(f)


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

    jobid: str
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
