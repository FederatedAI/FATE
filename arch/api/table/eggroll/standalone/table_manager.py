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

from typing import Iterable

from arch.api.standalone.eggroll import Standalone
from arch.api.table.eggroll.wrapped_dtable import DTable
from arch.api.table.table_manager import TableManager as TableManger


# noinspection PyProtectedMember
class DTableManager(TableManger):
    """
    manage RDDTable, use EggRoleStorage as storage
    """

    def __init__(self, job_id, eggroll_context):
        self._eggroll = Standalone(job_id=job_id, eggroll_context=eggroll_context)
        self.job_id = job_id
        TableManger.set_instance(self)

    def table(self,
              name,
              namespace,
              partition,
              persistent,
              in_place_computing,
              create_if_missing,
              error_if_exist
              ):
        dtable = self._eggroll.table(name=name,
                                     namespace=namespace,
                                     partition=partition,
                                     persistent=persistent,
                                     in_place_computing=in_place_computing,
                                     create_if_missing=create_if_missing,
                                     error_if_exist=error_if_exist)
        return DTable(dtable=dtable, job_id=self.job_id)

    def parallelize(self,
                    data: Iterable,
                    include_key,
                    name,
                    partition,
                    namespace,
                    persistent,
                    chunk_size,
                    in_place_computing,
                    create_if_missing,
                    error_if_exist):
        dtable = self._eggroll.parallelize(data=data,
                                           include_key=include_key,
                                           name=name,
                                           partition=partition,
                                           namespace=namespace,
                                           persistent=persistent,
                                           chunk_size=chunk_size,
                                           in_place_computing=in_place_computing,
                                           create_if_missing=create_if_missing,
                                           error_if_exist=error_if_exist)

        rdd_inst = DTable(dtable, job_id=self.job_id)

        return rdd_inst

    def cleanup(self, name, namespace, persistent):
        self._eggroll.cleanup(name=name, namespace=namespace, persistent=persistent)

    def generateUniqueId(self):
        self._eggroll.generateUniqueId()

    def get_job_id(self):
        return self.job_id
