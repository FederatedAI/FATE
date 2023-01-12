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
from ..conf.env_config import StandaloneConfig
from ..conf.types import ArtifactType
from ..entity.dag_structures import RuntimeTaskOutputChannelSpec
from ..entity.task_structure import TaskScheduleSpec, LOGGERSpec, TaskRuntimeInputSpec, \
    MLMDSpec, RuntimeConfSpec, ComputingEngineSpec, DeviceSpec, FederationPartySpec, \
    ComputingEngineMetadata, FederationEngineSpec, FederationEngineMetadata, InputArtifact
from ..manager.resource_manager import StandaloneResourceManager
from ..utils.standalone.id_gen import gen_computing_id, gen_federation_id, gen_task_id


class RuntimeConstructor(object):
    OUTPUT_KEYS = ["model", "metric", "data"]

    def __init__(self, runtime_parties, stage, job_id, task_name,
                 component_ref, component_spec, runtime_parameters, log_dir):
        self._task_name = task_name
        self._runtime_parties = runtime_parties
        self._job_id = job_id
        self._federation_id = gen_federation_id(job_id, task_name)
        self._stage = stage
        self._component_ref = component_ref
        self._component_spec = component_spec
        self._runtime_parameters = runtime_parameters
        self._log_dir = log_dir

        self._conf = StandaloneConfig()
        self._resource_manager = StandaloneResourceManager(self._conf)

        self._runtime_conf = dict()
        self._input_artifacts = dict()
        self._output_artifacts = dict()
        self._outputs = dict()
        self._task_schedule_spec = dict()
        self._task_conf_path = dict()

        self._runtime_roles = set()
        for party in self._runtime_parties:
            self._runtime_roles.add(party.role)
            if party.role not in self._runtime_conf:
                self._runtime_conf[party.role] = dict()
                self._input_artifacts[party.role] = dict()
                self._output_artifacts[party.role] = dict()
                self._outputs[party.role] = dict()
                self._task_schedule_spec[party.role] = dict()
                self._task_conf_path[party.role] = dict()

            self._runtime_conf[party.role].update({party.party_id: None})
            self._input_artifacts[party.role].update({party.party_id: dict()})
            self._output_artifacts[party.role].update({party.party_id: dict()})
            self._outputs[party.role].update({party.party_id: dict()})
            self._task_schedule_spec[party.role].update({party.party_id: None})
            self._task_conf_path[party.role].update({party.party_id: None})

    def construct_outputs(self):
        for output_key in self.OUTPUT_KEYS:
            for party in self._runtime_parties:
                output_artifact = self._construct_outputs(party, output_key)
                self._outputs[party.role][party.party_id].update({output_key: output_artifact})

    def _construct_outputs(self, party, artifact_type):
        return self._resource_manager.generate_output_artifact(
            self._job_id,
            self._task_name,
            party.role,
            party.party_id,
            artifact_type
        )

    def get_output_artifact(self, role, party_id, output_key):
        if role not in self._output_artifacts or party_id not in self._output_artifacts[role]:
            return None

        return self._output_artifacts[role][party_id].get(output_key, None)

    def construct_input_artifacts(self, upstream_inputs, runtime_constructor_dict,
                                  fit_model_info=None):
        input_artifacts = self._component_spec.input_definitions.artifacts
        for input_key, channels in upstream_inputs.items():
            artifact_spec = input_artifacts[input_key]
            if self._stage not in set(artifact_spec.stages):
                raise ValueError(f"Task stage is {self._stage}, not match input artifact's stage {artifact_spec.stage}")

            roles = set(channels[0].roles) if isinstance(channels, list) else set(channels.roles)
            optional = artifact_spec.optional
            artifact_type = artifact_spec.type
            for party in self._runtime_parties:
                if party.role not in roles:
                    continue

                if not isinstance(channels, list):
                    channels = [channels]

                output_artifacts = []
                for channel in channels:
                    upstream_task = channel.producer_task
                    upstream_output_key = channel.output_artifact_key

                    if isinstance(channel, RuntimeTaskOutputChannelSpec):
                        output_artifact = runtime_constructor_dict[upstream_task].get_output_artifact(
                            party.role, party.party_id, upstream_output_key)
                    else:
                        output_artifact = fit_model_info.task_info[upstream_task].get_output_artifact(
                            party.role, party.party_id, upstream_output_key
                        )

                    if output_artifact is None:
                        if not optional:
                            raise ValueError(f"Can not find upstream input {input_key} for "
                                             f"role={party.role}, party_id={party.party_id}, task={self._task_name}")
                        continue

                    output_artifacts.append(output_artifact)

                if artifact_type in ["dataset", "model"]:
                    output_artifacts = output_artifacts[0]

                self._input_artifacts[party.role][party.party_id].update({input_key: output_artifacts})

    def _construct_mlmd(self, role, party_id):
        metadata = {
            "db": self._conf.MLMD.db
        }
        return MLMDSpec(type=self._conf.MLMD.type,
                        metadata=metadata)

    def _construct_logger(self, role, party_id):
        metadata = dict(
            basepath=self._resource_manager.generate_log_uri(
                self._log_dir, role, party_id),
            level=self._conf.LOGGER.level,
            debug_mode=self._conf.LOGGER.debug_mode
        )
        return LOGGERSpec(type="pipeline",
                          metadata=metadata)

    def _construct_computing_engine(self, role, party_id):
        return ComputingEngineSpec(
            type=self._conf.COMPUTING_ENGINE.type,
            metadata=ComputingEngineMetadata(
                computing_id=gen_computing_id(self._job_id, self._task_name, role, party_id)
            )
        )

    def _construct_federation_engine(self, role, party_id):
        parties = []
        for party in self._runtime_parties:
            parties.append(dict(role=party.role, partyid=party.party_id))
        return FederationEngineSpec(
            type=self._conf.FEDERATION_ENGINE.type,
            metadata=FederationEngineMetadata(
                federation_id=self._federation_id,
                parties=FederationPartySpec(
                    local=dict(role=role, partyid=party_id),
                    parties=parties
                )
            )
        )

    def _construct_runtime_conf(self, role, party_id):
        mlmd = self._construct_mlmd(role, party_id)
        logger = self._construct_logger(role, party_id)
        computing_backend = self._construct_computing_engine(role, party_id)
        federation_backend = self._construct_federation_engine(role, party_id)
        return RuntimeConfSpec(
            mlmd=mlmd,
            logger=logger,
            device=DeviceSpec(type=self._conf.DEVICE.type),
            computing=computing_backend,
            federation=federation_backend,
            output=self._outputs[role][party_id]
        )

    def construct_task_schedule_spec(self):
        for party in self._runtime_parties:
            conf = self._construct_runtime_conf(party.role, party.party_id)
            party_task_spec = TaskScheduleSpec(
                task_id=self._federation_id,
                party_task_id=gen_task_id(self._job_id, self._task_name, party.role, party.party_id),
                component=self._component_ref,
                role=party.role,
                party_id=party.party_id,
                stage=self._stage,
                conf=conf
            )

            input_artifact = self._input_artifacts[party.role][party.party_id]
            task_input_spec = TaskRuntimeInputSpec()
            if input_artifact:
                task_input_spec.artifacts = input_artifact
            parameters = self._runtime_parameters.get(party.role, {}).get(party.party_id, {})
            if parameters:
                task_input_spec.parameters = parameters

            if task_input_spec.dict(exclude_defaults=True):
                party_task_spec.inputs = task_input_spec

            # output_artifact = self._output_artifacts[party.role][party.party_id]
            # if output_artifact:
            #     party_task_spec.outputs = TaskRuntimeOutputSpec(artifacts=output_artifact)

            self._task_schedule_spec[party.role][party.party_id] = party_task_spec
            conf_path = self._resource_manager.write_out_task_conf(self._job_id,
                                                                   self._task_name,
                                                                   party.role,
                                                                   party.party_id,
                                                                   party_task_spec.dict(exclude_defaults=True))
            self._task_conf_path[party.role][party.party_id] = conf_path
            print(f"{party.role}-{party.party_id}'s task_conf path {conf_path}")

    @property
    def runtime_parties(self):
        return self._runtime_parties

    def mlmd(self, role, party_id):
        return self._task_schedule_spec[role][party_id].conf.mlmd

    def task_conf_path(self, role, party_id):
        return self._task_conf_path[role][party_id]

    def party_task_id(self, role, party_id):
        return self._task_schedule_spec[role][party_id].party_task_id

    @property
    def status_manager(self):
        return self._resource_manager.status_manager

    def log_path(self, role, party_id):
        return self._task_schedule_spec[role][party_id].conf.logger.metadata["basepath"]

    def retrieval_task_outputs(self):
        for party in self._runtime_parties:
            party_task_id = self._task_schedule_spec[party.role][party.party_id].party_task_id
            outputs = self._resource_manager.status_manager.get_task_outputs(party_task_id)

            for output_key in self.OUTPUT_KEYS:
                output_list = outputs.get(output_key)
                if not output_list:
                    continue

                for output in output_list:
                    output_artifact = InputArtifact(**output)
                    self._output_artifacts[party.role][party.party_id].update({output_artifact.name: output_artifact})

    def get_output_data(self, role, party_id):
        data = dict()
        for artifact_key, artifact in self._output_artifacts[role][party_id].items():
            artifact_spec = self._component_spec.output_definitions.artifacts[artifact_key]
            uri = artifact.uri
            if artifact_spec.type in [ArtifactType.DATASET, ArtifactType.DATASETS]:
                data[artifact_key] = self._resource_manager.get_output_data(uri)

        return data

    def get_output_model(self, role, party_id):
        models = dict()
        for artifact_key, artifact in self._output_artifacts[role][party_id].items():
            artifact_spec = self._component_spec.output_definitions.artifacts[artifact_key]
            uri = artifact.uri
            if artifact_spec.type in [ArtifactType.MODEL, ArtifactType.MODELS]:
                models.update(self._resource_manager.get_output_model(uri))

        return models

    def get_output_metrics(self, role, party_id):
        metrics = dict()
        for artifact_key, artifact in self._output_artifacts[role][party_id].items():
            artifact_spec = self._component_spec.output_definitions.artifacts[artifact_key]
            uri = artifact.uri
            if ArtifactType.METRIC in artifact_spec.type:
                metric_name = uri.split("/", -1)[-1]
                metrics[metric_name] = self._resource_manager.get_output_metrics(uri)

        return metrics


