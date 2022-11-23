from ..conf.env_config import StandaloneConfig
from ..entity.component_structures import OutputDefinitionsSpec
from ..entity.task_structure import TaskScheduleSpec, MLMDSpec, LOGGERSpec, \
    RuntimeEnvSpec, ComputingBackendSpec, FederationBackendSpec, FederationPartySpec
from ..manager.resource_manager import StandaloneResourceManager
from ..utils.id_gen import gen_computing_id, gen_federation_id, gen_execution_id


class RuntimeConstructor(object):
    def __init__(self, runtime_parties, stage, job_id, task_name, component_ref, runtime_parameters, log_dir):
        self._task_name = task_name
        self._runtime_parties = runtime_parties
        self._job_id = job_id
        self._federation_id = gen_federation_id(job_id, task_name)
        self._stage = stage
        self._component_ref = component_ref
        self._runtime_parameters = runtime_parameters
        self._log_dir = log_dir

        self._conf = StandaloneConfig()
        self._resource_manager = StandaloneResourceManager(self._conf)

        self._runtime_env = dict()
        self._runtime_conf = dict()
        self._input_artifacts = dict()
        self._output_artifacts = dict()
        self._task_schedule_spec = dict()
        self._task_conf_uri = dict()

        self._runtime_roles = set()
        for party in self._runtime_parties:
            self._runtime_roles.add(party.role)
            if party.role not in self._runtime_conf:
                self._runtime_conf[party.role] = dict()
                self._input_artifacts[party.role] = dict()
                self._output_artifacts[party.role] = dict()
                self._task_schedule_spec[party.role] = dict()
                self._task_conf_uri[party.role] = dict()

            self._runtime_conf[party.role].update({party.party_id: None})
            self._input_artifacts[party.role].update({party.party_id: dict()})
            self._output_artifacts[party.role].update({party.party_id: dict()})
            self._task_schedule_spec[party.role].update({party.party_id: None})
            self._task_conf_uri[party.role].update({party.party_id: None})

    def construct_output_artifacts(self, output_definition_artifacts: OutputDefinitionsSpec):
        for output_key, output_artifact_spec in output_definition_artifacts.artifacts.items():
            if self._stage not in set(output_artifact_spec.stages):
                continue
            roles = set(output_artifact_spec.roles)

            for party in self._runtime_parties:
                if party.role not in roles:
                    continue

                output_artifact = self._construct_output_artifact(party, output_key, output_artifact_spec.type)

                if output_artifact:
                    self._output_artifacts[party.role][party.party_id].update({output_key: output_artifact})

    def _construct_output_artifact(self, party, output_key,  artifact_type):
        return self._resource_manager.generate_output_artifact(
            self._job_id,
            self._task_name,
            party.role,
            party.party_id,
            output_key,
            artifact_type
        )

    def get_output_artifact(self, role, party_id, output_key):
        if role not in self._output_artifacts or party_id not in self._output_artifacts[role]:
            return None

        return self._output_artifacts[role][party_id].get(output_key, None)

    def construct_input_artifacts(self, upstream_inputs, runtime_constructor_dict, component_spec):
        input_artifacts = component_spec.input_definitions.artifacts
        for input_key, channels in upstream_inputs.items():
            artifact_spec = input_artifacts[input_key]
            if self._stage not in set(artifact_spec.stages):
                raise ValueError(f"Task stage is {self._stage}, not match input artifact's stage {artifact_spec.stage}")

            roles = set(artifact_spec.roles)
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

                    output_artifact = runtime_constructor_dict[upstream_task].get_output_artifact(
                        party.role, party.party_id, upstream_output_key)
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
        status_path = self._resource_manager.generate_output_status_uri(self._job_id, self._task_name, role, party_id)
        terminate_status_path = self._resource_manager.generate_output_terminate_status_uri(
            self._job_id, self._task_name, role, party_id)
        metadata = {
            "state_path": status_path,
            "terminate_state_path": terminate_status_path
        }
        return MLMDSpec(type="pipeline",
                        metadata=metadata)

    def _construct_logger(self, role, party_id):
        metadata = dict(
            base_path=self._resource_manager.generate_log_uri(
                self._log_dir, self._job_id, self._task_name, role, party_id),
            level=self._conf.LOG_LEVEL,
            debug_mode=self._conf.LOG_DEBUG_MODE
        )
        return LOGGERSpec(type="pipeline",
                          metadata=metadata)

    def _construct_computing_backend(self, role, party_id):
        return ComputingBackendSpec(
            engine="standalone",
            computing_id=gen_computing_id(self._job_id, self._task_name, role, party_id)
        )

    def _construct_federation_backend(self, role, party_id):
        parties = []
        for party in self._runtime_parties:
            parties.append(dict(role=party.role, partyid=party.party_id))
        return FederationBackendSpec(
            engine="standalone",
            federation_id=self._federation_id,
            parties=FederationPartySpec(
                local=dict(role=role, partyid=party_id),
                parties=parties
            )
        )

    def _construct_runtime_env(self, role, party_id):
        mlmd = self._construct_mlmd(role, party_id)
        logger = self._construct_logger(role, party_id)
        computing_backend = self._construct_computing_backend(role, party_id)
        federation_backend = self._construct_federation_backend(role, party_id)
        return RuntimeEnvSpec(
            mlmd=mlmd,
            logger=logger,
            device=self._conf.DEVICE,
            distributed_computing_backend=computing_backend,
            federation_backend=federation_backend
        )

    def construct_task_schedule_spec(self):
        for party in self._runtime_parties:
            env = self._construct_runtime_env(party.role, party.party_id)
            party_task_spec = TaskScheduleSpec(
                execution_id=gen_execution_id(self._job_id, self._task_name, party.role, party.party_id),
                component=self._component_ref,
                role=party.role,
                stage=self._stage,
                env=env
            )

            input_artifact = self._input_artifacts[party.role][party.party_id]
            if input_artifact:
                party_task_spec.inputs = input_artifact

            output_artifact = self._output_artifacts[party.role][party.party_id]
            if output_artifact:
                party_task_spec.outputs = output_artifact

            self._task_schedule_spec[party.role][party.party_id] = party_task_spec
            conf_uri = self._resource_manager.write_out_task_conf(self._job_id,
                                                                  self._task_name,
                                                                  party.role,
                                                                  party.party_id,
                                                                  party_task_spec.dict(exclude_defaults=True))
            self._task_conf_uri[party.role][party.party_id] = conf_uri
            print(f"{party.role}-{party.party_id}'s task_conf uri {conf_uri}")

    @property
    def runtime_parties(self):
        return self._runtime_parties

    def mlmd(self, role, party_id):
        return self._task_schedule_spec[role][party_id].env.mlmd

    def task_conf_uri(self, role, party_id):
        return self._task_conf_uri[role][party_id]

    def execution_id(self, role, party_id):
        return self._task_schedule_spec[role][party_id].execution_id

    @property
    def status_manager(self):
        return self._resource_manager.status_manager
