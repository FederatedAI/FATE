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

import uuid
from typing import Iterable

from arch.api.core import EggRollContext
from arch.api.table.pyspark import materialize
from arch.api.table.pyspark.cluster.rddtable import RDDTable
from arch.api.table.table_manager import TableManager
from arch.api.utils import file_utils


# noinspection PyProtectedMember
class RDDTableManager(TableManager):
    """
    manage RDDTable, use EggRoleStorage as storage
    """

    def __init__(self, job_id, eggroll_context, server_conf_path="arch/conf/server_conf.json", master=None):

        self.job_id = job_id
        self._eggroll_context = eggroll_context
        self._init_eggroll(server_conf_path)

        # init PySpark
        from pyspark import SparkContext, SparkConf
        conf = SparkConf().setAppName(f"FATE-PySpark-{job_id}")
        if master:
            conf = conf.setMaster(master)
        sc = SparkContext.getOrCreate(conf=conf)
        self._sc = sc

        # set eggroll info
        import pickle
        from arch.api.table.pyspark import _EGGROLL_CLIENT
        pickled_client = pickle.dumps(dict(job_id=self.job_id,
                                           host=self._roll_host,
                                           port=self._roll_port,
                                           eggroll_context=self._eggroll_context)).hex()
        sc.setLocalProperty(_EGGROLL_CLIENT, pickled_client)
        TableManager.set_instance(self)

    def _init_eggroll(self, server_conf_path):
        """
        modified from :func:`arch.api.cluster.eggroll.init`
        """
        if self.job_id is None:
            self.job_id = str(uuid.uuid1())
        server_conf = file_utils.load_json_conf(server_conf_path)
        self._roll_host = server_conf.get("servers").get("roll").get("host")
        self._roll_port = server_conf.get("servers").get("roll").get("port")

        if self._eggroll_context is None:
            self._eggroll_context = EggRollContext()

        from arch.api.cluster.eggroll import init
        from arch.api.cluster.eggroll import _EggRoll
        if _EggRoll.instance is None:
            init(self.job_id, server_conf_path=server_conf_path, eggroll_context=self._eggroll_context)
        self._eggroll: _EggRoll = _EggRoll.instance

    def table(self,
              name,
              namespace,
              partition,
              persistent,
              in_place_computing,
              create_if_missing,
              error_if_exist):
        dtable = self._eggroll.table(name=name, namespace=namespace, partition=partition,
                                     persistent=persistent, in_place_computing=in_place_computing,
                                     create_if_missing=create_if_missing, error_if_exist=error_if_exist)
        return RDDTable.from_dtable(job_id=self.job_id, dtable=dtable)

    def parallelize(self,
                    data: Iterable,
                    name,
                    namespace,
                    partition,
                    include_key,
                    persistent,
                    chunk_size,
                    in_place_computing,
                    create_if_missing,
                    error_if_exist):
        _iter = data if include_key else enumerate(data)
        rdd = self._sc.parallelize(_iter, partition)
        rdd = materialize(rdd)
        if namespace is None:
            namespace = self.job_id
        return RDDTable.from_rdd(rdd=rdd, job_id=self.job_id, namespace=namespace, name=name)

    def cleanup(self, name, namespace, persistent):
        return self._eggroll.cleanup(name=name, namespace=namespace, persistent=persistent)

    def generateUniqueId(self):
        return self._eggroll.generateUniqueId()
