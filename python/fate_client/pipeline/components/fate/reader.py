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
from typing import List
from ...conf.types import PlaceHolder
from ..component_base import Component


class Reader(Component):
    yaml_define_path = "./component_define/fate/reader.yaml"

    def __init__(self,
                 name: str,
                 runtime_roles: List[str] = None,
                 path: str = PlaceHolder(),
                 format: str = PlaceHolder(),
                 id_name: str = PlaceHolder(),
                 delimiter: str = PlaceHolder(),
                 label_name: str = PlaceHolder(),
                 label_type: str = PlaceHolder(),
                 dtype: str = PlaceHolder()
                 ):
        inputs = locals()
        self._process_init_inputs(inputs)
        super(Reader, self).__init__()
        self.name = name
        self.runtime_roles = runtime_roles
        self.path = path
        self.format = format
        self.id_name = id_name
        self.delimiter = delimiter
        self.label_name = label_name
        self.label_type = label_type
        self.dtype = dtype
