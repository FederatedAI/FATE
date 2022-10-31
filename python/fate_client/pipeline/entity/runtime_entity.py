import pprint
from types import SimpleNamespace
from ..conf.env_config import FateStandaloneConfig
from ..utils.id_gen import get_session_id, get_computing_id, get_federation_id
from ..manager.resource_manager import FateStandaloneResourceManager


class Roles(object):
    def __init__(self):
        self._role_party_mappings = dict()
        self._role_party_index_mapping = dict()
        self._leader_role = None

    def set_role(self, role, party_id):
        if not isinstance(party_id, list):
            party_id = [party_id]

        if role not in self._role_party_mappings:
            self._role_party_mappings[role] = []
            self._role_party_index_mapping[role] = dict()

        for pid in party_id:
            if pid in self._role_party_index_mapping[role]:
                raise ValueError(f"role {role}, party {pid} has been added before")
            self._role_party_index_mapping[role][pid] = len(self._role_party_mappings[role])
            self._role_party_mappings[role].append(pid)

        self._role_party_mappings[role] = party_id

    def set_leader(self, role, party_id):
        self._leader_role = SimpleNamespace(role=role,
                                            party_id=party_id)

    @property
    def leader(self):
        return self._leader_role

    def get_party_list_by_role(self, role):
        return self._role_party_mappings[role]

    def get_party_by_role_index(self, role, index):
        return self._role_party_mappings[role][index]

    def get_runtime_roles(self):
        return self._role_party_mappings.keys()


class FateStandaloneRuntimeEntity(object):
    def __init__(self, session_id, node_conf, node_name, module, job_type,
                 runtime_parties, inputs=None, outputs=None):
        """
        session_id: runtime session_id, should be unique for every task
        node_conf: parameters of the task
        node_name: component name
        inputs: input links dict
        outputs: an object, can be retrieval, use to generate output uri for running task
        """
        self._session_id_prefix = session_id
        self._node_conf = node_conf
        self._node_name = node_name
        self._module = module
        self._job_type = job_type
        self._runtime_parties = runtime_parties
        self._runtime_conf = dict()
        self._runtime_role_with_party = list()
        self._inputs = inputs
        self._outputs = outputs

        self._resource_manager = None

        if self._module == "data_input":
            self._prepare_local_data_input()
        else:
            self._prepare_runtime_env()

    def _prepare_runtime_env(self):
        conf = FateStandaloneConfig()
        resource_manager = FateStandaloneResourceManager(conf)
        session_id = get_session_id(self._session_id_prefix, self._node_name)
        computing_id = get_computing_id(session_id)
        federation_id = get_federation_id(session_id)

        # TODO: backend construct
        backends = dict(device=conf.DEVICE,
                        computing_engine=conf.COMPUTING_ENGINE,
                        federation_engine=conf.FEDERATION_ENGINE)

        for role, role_conf in self._node_conf.items():
            self._runtime_conf[role] = dict()
            for party, party_conf in role_conf.items():
                self._runtime_role_with_party.append(SimpleNamespace(role=role, party_id=party))
                outputs = resource_manager.generate_task_outputs(session_id=session_id,
                                                                 role=role,
                                                                 party=party,
                                                                 outputs=self._outputs)
                inputs = self._generate_inputs(role, party)
                job_conf_uri = resource_manager.generate_job_conf_uri(session_id=session_id,
                                                                      role=role,
                                                                      party=party)

                print(f"{role}-{party}'s job_conf_uri: {job_conf_uri}, outputs is ")
                pprint.pprint(outputs)
                print("\n\n")

                party_runtime_conf = resource_manager.generate_task_runtime_conf(
                    session_id=session_id,
                    computing_id=computing_id,
                    federation_id=federation_id,
                    job_type=self._job_type,
                    module=self._module,
                    runtime_parties=self._runtime_parties,
                    role=role,
                    party=party,
                    params=party_conf,
                    inputs=inputs,
                    outputs=outputs,
                    backends=backends,
                )

                resource_manager.write_out_job_conf(job_conf_uri, party_runtime_conf)
                self._runtime_conf[role][party] = SimpleNamespace(job_conf_uri=job_conf_uri,
                                                                  conf=party_runtime_conf)

                # import pprint
                # pprint.pprint(party_runtime_conf)
                # pprint.pprint(party_conf)
                # print(session_id, self._session_id_prefix)
                # pprint.pprint(self._runtime_parties)
                # pprint.pprint(inputs)

        self._resource_manager = resource_manager

    def _prepare_local_data_input(self):
        # TODO: local data input is not same with other components, just for mini-demo
        for role, role_conf in self._node_conf.items():
            self._runtime_conf[role] = dict()
            for party, party_conf in role_conf.items():
                party_runtime_conf = dict(
                    task=dict(
                        task_params=dict(
                            data_outputs=list(party_conf.items())[0][1]
                )))
                self._runtime_conf[role][party] = SimpleNamespace(
                    conf=party_runtime_conf
                )
                self._runtime_role_with_party.append(SimpleNamespace(role=role, party_id=party))

    def _generate_inputs(self, role, party):
        inputs = dict()

        """
        inputs = {
            "model": {
                "src_i": {
                    (role, party_id): path
                }
            },
            ...
        }
        """
        for outer_input_key, outer_inputs in self._inputs.items():
            for inner_input_key, upstream_inputs in outer_inputs.items():
                for src, role_party_path in upstream_inputs.items():
                    if (role, party) in role_party_path:
                        inputs[inner_input_key] = dict()

                        inputs[inner_input_key].update({
                            src: role_party_path[(role, party)]
                        })

        return inputs

    @property
    def runtime_role_with_party(self):
        return self._runtime_role_with_party

    """
    def get_log_path(self, role, party_id):
        return self._runtime_conf[role][party_id]["output"]["log"]
    """

    def get_job_conf_uri(self, role, party_id):
        return self._runtime_conf[role][party_id].job_conf_uri

    def get_data_output_uri(self, role=None, party_id=None, key=None):
        """
        model and data is diff, so use two different function instead,
        """
        #TODO: data many has many outputs instead of one, like data splits
        if role is not None:
            return self._runtime_conf[role][party_id].party_runtime_conf["outputs"]["data"]
        else:
            ret = dict()
            for role_with_party in self._runtime_role_with_party:
                role = role_with_party.role
                party_id = role_with_party.party_id
                ret[(role, party_id)] = self._runtime_conf[role][party_id].conf["task"]["task_params"]["data_outputs"]

            return ret

    def get_model_output_uri(self, role=None, party_id=None, key=None):
        """
        model and data is diff, so use two different function instead
        """
        if role is not None:
            return self._runtime_conf[role][party_id].conf["outputs"]["model"]
        else:
            ret = dict()
            for role_with_party in self._runtime_role_with_party:
                role = role_with_party.role
                party_id = role_with_party.party_id
                ret[(role, party_id)] = self._runtime_conf[role][party_id].conf["task"]["task_params"]["model_outputs"]

            return ret

    def get_metric_output_uri(self, role, party_id):
        return self._runtime_conf[role][party_id].conf["task"]["task_params"]["metrics_output"]

    def get_status_output_uri(self, role, party_id):
        return self._runtime_conf[role][party_id].conf["task"]["task_params"]["status_output"]

    @property
    def status_manager(self):
        return self._resource_manager.status_manager
