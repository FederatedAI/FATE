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

import hashlib
from arch.api.utils import log_utils, version_control
from federatedml.model_base import ModelBase
from federatedml.param.rsa_param import RsaParam
from federatedml.secureprotol import gmpy_math
from arch.api import session
import datetime

LOGGER = log_utils.getLogger()


class RsaModel(ModelBase):
    """
    encrypt data using RSA

    Parameters
    ----------
    RsaParam : object, self-define id_process parameters,
        define in federatedml.param.rsa_param

    """
    def __init__(self):
        super(RsaModel, self).__init__()
        self.data_processed = None
        self.model_param = RsaParam()


    def fit(self, data_inst):
        LOGGER.info("RsaModel start fit...")
        LOGGER.debug("data_inst={}, count={}".format(data_inst, data_inst.count()))

        key_pair = {"d": self.model_param.rsa_key_d, "n": self.model_param.rsa_key_n}
        self.data_processed = self.encrypt_data_using_rsa(data_inst, key_pair)

        self.save_data()

    def save_data(self):
        #LOGGER.debug("save data: data_inst={}, count={}".format(self.data_processed, self.data_processed.count()))
        persistent_table = self.data_processed.save_as(namespace=self.model_param.save_out_table_namespace, name=self.model_param.save_out_table_name)
        LOGGER.info("save data to namespace={}, name={}".format(persistent_table._namespace, persistent_table._name))
    
        session.save_data_table_meta(
            {'schema': self.data_processed.schema, 'header': self.data_processed.schema.get('header', [])},
            data_table_namespace=persistent_table._namespace, data_table_name=persistent_table._name)
        
        version_log = "[AUTO] save data at %s." % datetime.datetime.now()
        version_control.save_version(name=persistent_table._name, namespace=persistent_table._namespace, version_log=version_log)
        return None


    @staticmethod
    def hash(value):
        return hashlib.sha256(bytes(str(value), encoding='utf-8')).hexdigest()


    def encrypt_data_using_rsa(self, data_inst, key_pair):
        LOGGER.info("encrypt data using rsa: {}".format(str(key_pair)))
        data_processed_pair = data_inst.map(
            lambda k, v: (
                RsaModel.hash(gmpy_math.powmod(int(RsaModel.hash(k), 16), key_pair["d"], key_pair["n"])), k)
        )

        return data_processed_pair

