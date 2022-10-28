from types import SimpleNamespace
from ..conf.env_config import FateStandaloneConfig
from ..utils.id_gen import get_session_id
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
    def __init__(self, session_id, node_conf, node_name, inputs=None, outputs=None):
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
        self._runtime_conf = dict()
        self._runtime_role_with_party = list()
        self._inputs = inputs
        self._outputs = outputs

        self._resource_manager = None

        self._prepare_runtime_env()

    def _prepare_runtime_env(self):
        conf = FateStandaloneConfig()
        resource_manager = FateStandaloneResourceManager(conf)
        session_id = get_session_id(self._session_id_prefix, self._node_name)

        for role, role_conf in self._node_conf.items():
            self._runtime_conf[role] = dict()
            for party, party_conf in role_conf.items():
                self._runtime_role_with_party.append(SimpleNamespace(role=role, party_id=party))
                party_runtime_conf = dict(param=party_conf)
                outputs = resource_manager.generate_task_runtime_resource(session_id=session_id,
                                                                          role=role,
                                                                          party=party,
                                                                          outputs=self._outputs)

                party_runtime_conf["outputs"] = outputs
                party_runtime_conf = self._add_inputs(role, party, party_runtime_conf)
                job_conf_uri = resource_manager.generate_job_conf_uri(session_id=session_id,
                                                                      role=role,
                                                                      party=party)
                resource_manager.write_out_job_conf(job_conf_uri, party_runtime_conf)
                self._runtime_conf[role][party] = SimpleNamespace(job_conf_uri=job_conf_uri,
                                                                  conf=party_runtime_conf)

        self._resource_manager = resource_manager

    """
    def _prepare(self):
        conf = FateStandaloneConfig

        namespace = "output_data"
        name = get_uuid()
        for role, role_conf in self._node_conf.items():
            self._runtime_conf[role] = dict()
            for party, party_conf in role_conf.items():
                self._runtime_role_with_party.append(SimpleNamespace(role=role, party_id=party))
                party_runtime_conf = dict(param=party_conf)
                output_data_path = generate_dir(conf.OUTPUT_DATA_DIR, self._session_id,
                                                role, party, self._node_name, namespace, name).as_uri()
                output_model_path = generate_dir(conf.OUTPUT_MODEL_DIR, self._session_id, role, party,
                                                 self._node_name).as_uri()
                output_metric_path = generate_dir(conf.OUTPUT_METRIC_DIR, self._session_id, role, party,
                                                  self._node_name).as_uri()
                output_log_path = generate_dir(conf.LOG_DIR, self._session_id,
                                               role, party, self._node_name, "log").as_uri()
                output_status_path = generate_dir(conf.LOG_DIR, self._session_id, role,
                                                  party, self._node_name, "status", "status.log").as_uri()

                job_conf_path = generate_dir(conf.JOB_DIR, self._session_id, role,
                                             party, self._node_name, "runtime_conf.json")

                party_runtime_conf["output"] = dict(
                    data=output_data_path,
                    model=output_model_path,
                    metric=output_metric_path,
                    log=output_log_path,
                    status=output_status_path
                )
                party_runtime_conf["job_conf_path"] = job_conf_path.as_uri()
                party_runtime_conf = self._add_inputs(role, party, party_runtime_conf)
                write_json_file(str(job_conf_path), party_runtime_conf)

                self._runtime_conf[role][party] = party_runtime_conf
    """

    def _add_inputs(self, role, party, runtime_conf):
        if not self._inputs:
            return runtime_conf

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
        for input_key, upstream_inputs in self._inputs.items():
            for src, role_party_path in upstream_inputs.items():
                if (role, party) in role_party_path:
                    if "input" not in runtime_conf:
                        runtime_conf["input"] = dict()

                    if input_key not in runtime_conf["input"]:
                        runtime_conf["input"][input_key] = dict()

                    runtime_conf["input"][input_key].update({
                        src: role_party_path
                    })

        return runtime_conf

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
                ret[(role, party_id)] = self._runtime_conf[role][party_id].conf["outputs"]["data"]

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
                ret[(role, party_id)] = self._runtime_conf[role][party_id].conf["outputs"]["model"]

            return ret

    def get_metric_output_uri(self, role, party_id):
        return self._runtime_conf[role][party_id].conf["outputs"]["metric"]

    def get_status_output_uri(self, role, party_id):
        return self._runtime_conf[role][party_id].conf["outputs"]["status"]

    @property
    def status_manager(self):
        return self._resource_manager.status_manager
