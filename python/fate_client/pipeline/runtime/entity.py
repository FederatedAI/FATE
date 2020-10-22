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
#
import copy
from pipeline.utils.tools import extract_explicit_parameter


class JobParameters(object):
    @extract_explicit_parameter
    def __init__(self, work_mode=0, job_type="train", backend=0, computing_engine=None, federation_engine=None,
                 storage_engine=None, engines_address=None,federated_mode=None, federation_info=None, task_parallelism=None,
                 federated_status_collect_type=None, federated_data_exchange_type=None, model_id=None, model_version=None,
                 dsl_version=None, timeout=None, eggroll_run=None, spark_run=None, adaptation_parameters=None, **kwargs):
        explicit_parameters = kwargs["explict_parameters"]
        for param_key, param_value in explicit_parameters.items():
            setattr(self, param_key, param_value)

        self.__party_instance = {}
        self._job_param = {}

    def get_party_instance(self, role="guest", party_id=None):
        if role not in ["guest", "host", "arbiter"]:
            raise ValueError("Role should be one of guest/host/arbiter")

        if party_id is not None:
            if isinstance(party_id, list):
                for _id in party_id:
                    if not isinstance(_id, int) or _id <= 0:
                        raise ValueError("party id should be positive integer")
            elif not isinstance(party_id, int) or party_id <= 0:
                raise ValueError("party id should be positive integer")

        if role not in self.__party_instance:
            self.__party_instance[role] = {}
            self.__party_instance[role]["party"] = {}

        party_key = party_id

        if isinstance(party_id, list):
            party_key = "|".join(map(str, party_id))

        if party_key not in self.__party_instance[role]["party"]:
            self.__party_instance[role]["party"][party_key] = None

        if not self.__party_instance[role]["party"][party_key]:
            party_instance = copy.deepcopy(self)

            self.__party_instance[role]["party"][party_key] = party_instance

        return self.__party_instance[role]["party"][party_key]

    def job_param(self, **kwargs):
        new_kwargs = copy.deepcopy(kwargs)
        for attr in new_kwargs:
            setattr(self, attr, new_kwargs[attr])
            self._job_param[attr] = new_kwargs[attr]

    def get_job_param(self):
        return self._job_param

    def get_common_param_conf(self):
        common_param_conf = {}
        for attr in self.__dict__:
            if attr.startswith("_"):
                continue

            common_param_conf[attr] = getattr(self, attr)

        return common_param_conf

    def get_role_param_conf(self, roles=None):
        role_param_conf = {}

        if not self.__party_instance:
            return role_param_conf

        for role in self.__party_instance:
            role_param_conf[role] = {}
            if None in self.__party_instance[role]["party"]:
                role_all_party_conf = self.__party_instance[role]["party"][None].get_job_param()
                if "all" not in role_param_conf:
                    role_param_conf[role]["all"] = {}
                    role_param_conf[role]["all"] = role_all_party_conf

            valid_partyids = roles.get(role)
            for party_id in self.__party_instance[role]["party"]:
                if not party_id:
                    continue

                if isinstance(party_id, int):
                    party_key = str(valid_partyids.index(party_id))
                else:
                    party_list = list(map(int, party_id.split("|", -1)))
                    party_key = "|".join(map(str, [valid_partyids.index(party) for party in party_list]))

                party_inst = self.__party_instance[role]["party"][party_id]

                if party_key not in role_param_conf:
                    role_param_conf[role][party_key] = {}

                role_param_conf[role][party_key] = party_inst.get_job_param()

        return role_param_conf

    def get_config(self, *args, **kwargs):
        """need to implement"""

        roles = kwargs["roles"]

        common_param_conf = self.get_common_param_conf()
        role_param_conf = self.get_role_param_conf(roles)

        conf = {}
        if common_param_conf:
            conf['common'] = common_param_conf

        if role_param_conf:
            conf["role"] = role_param_conf

        return conf


