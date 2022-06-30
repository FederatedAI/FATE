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
# from fate_arch.storage import DEFAULT_ID_DELIMITER
from fate_arch.storage._types import DEFAULT_ID_DELIMITER
from federatedml.param.base_param import BaseParam

class UploadFromDBParam(BaseParam):
    def __init__(self,
                 db_type="",
                 table="",
                 host="",
                 id_delimiter=DEFAULT_ID_DELIMITER,
                 port=0,
                 user="",
                 passwd="",
                 db="",
                 id="",
                 params=[],
                 namespace="",
                 name="",
                 partition=16,
                 destroy=True,
                 storage_engine='',
                 storage_address=''):
        super(UploadFromDBParam, self).__init__()
        self.db_type = db_type
        self.table = table
        self.host = host
        self.id_delimiter = id_delimiter
        self.port = port
        self.passwd = passwd
        self.db = db
        self.id = id  # id 主键
        self.params = params  # 需要获取的字段名
        self.namespace = namespace
        self.name = name
        self.destroy = destroy
        self.user = user
        self.partition = partition
        self.storage_engine = storage_engine
        self.storage_address = storage_address

    def check(self):
       if self.db_type == '' or self.host == '' or self.port == '' or self.passwd == '' or self.user == '' or self.table == '':
           raise ValueError('db infomation cannot be empty')
