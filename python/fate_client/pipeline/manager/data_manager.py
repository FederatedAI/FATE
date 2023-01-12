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
import json
import pandas as pd
from typing import Union
from ..utils.uri_tools import parse_uri, replace_uri_path, get_schema_from_uri
from ..utils.file_utils import construct_local_dir
from ..conf.types import UriTypes


class LocalFSDataManager(object):
    @classmethod
    def generate_output_data_uri(cls, output_dir_uri: str, job_id: str, task_name: str,
                                 role: str, party_id: Union[str, int]):
        uri_obj = parse_uri(output_dir_uri)
        namespace = "_".join([job_id, task_name, role, str(party_id)])
        local_path = construct_local_dir(uri_obj.path, *[namespace])
        uri_obj = replace_uri_path(uri_obj, str(local_path))
        return uri_obj.geturl()

    @classmethod
    def get_output_data(cls, uri):
        uri_obj = parse_uri(uri)
        data = None
        schema = None
        with open(uri_obj.path, "r") as fin:
            for k, v in json.loads(fin.read()):
                if not data:
                    data = v
                else:
                    data.extend(v)

        with open(uri_obj.path + ".meta", "r") as fin:
            schema = json.loads(fin.read())

        columns = [field["name"] for field in schema["fields"]]

        df = pd.DataFrame(data, columns=columns)
        df = df.set_index(columns[0])

        return df


def get_data_manager(output_dir_uri):
    uri_type = get_schema_from_uri(output_dir_uri)
    if uri_type == UriTypes.LOCAL:
        return LocalFSDataManager
