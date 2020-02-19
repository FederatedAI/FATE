#!/usr/bin/env python    
# -*- coding: utf-8 -*- 

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

# =============================================================================
# Transfer Variable Generator.py
# =============================================================================

import json
import os
from collections import OrderedDict

from arch.api.utils import file_utils

BASE_DIR = os.path.join(file_utils.get_project_base_directory(), "federatedml", "transfer_variable")
TEMPLATE = os.path.join(BASE_DIR, "transfer_variable.template")
CONF_DIR = os.path.join(BASE_DIR, "definition")
MERGE_CONF_FILE_NAME = "transfer_conf.json"
MERGE_CONF_PATH = os.path.join(CONF_DIR, MERGE_CONF_FILE_NAME)
CLASS_DIR = os.path.join(BASE_DIR, "transfer_class")

SPACES = "    "


def write_out_class(writer, class_name, variable_names):
    def create_variable(name):
        return f"self.{name} = self._create_variable(name='{name}')"

    with open(TEMPLATE, "r") as fin:
        temp = fin.read()
        temp = temp.format(class_name=class_name,
                           create_variable=f"\n{SPACES}{SPACES}".join(map(create_variable, variable_names)))
        writer.write(temp)
    writer.flush()


def generate():
    merge_dict = OrderedDict()  # makes merged conf in order

    for f_name in sorted(os.listdir(CONF_DIR)):
        if not f_name.endswith(".json") or f_name == MERGE_CONF_FILE_NAME:
            continue

        with open(os.path.join(CONF_DIR, f_name), "r") as fin:
            var_dict = json.loads(fin.read())
            merge_dict.update(var_dict)

        name = f_name.split(".")[0]
        class_save_path = os.path.join(CLASS_DIR, f"{name}_transfer_variable.py")
        with open(class_save_path, "w") as f:
            keys = list(var_dict.keys())
            assert len(keys) == 1, "multi class defined in a single json"
            class_name = keys[0]
            variable_names = sorted(var_dict.get(class_name).keys())
            write_out_class(f, class_name, variable_names)

    # save a merged transfer variable conf, for federation auth checking.
    with open(MERGE_CONF_PATH, "w") as f:
        json.dump(merge_dict, f, indent=1)


if __name__ == "__main__":
    generate()
