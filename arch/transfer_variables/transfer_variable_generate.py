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

import argparse
import json
import os
import sys

from arch.api.utils import file_utils

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transfer_variable.template")
SPACES = "    "
DEFAULT_DST = os.path.join(file_utils.get_project_base_directory(), "federatedml", "transfer_variable",
                           "transfer_class")
DEFAULT_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auth_conf", "federatedml")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate transfer class from conf")
    parser.add_argument("-s", "--src", type=str, default=DEFAULT_SRC,
                        help=f"path or directory of json file for class generation, defaults to {DEFAULT_SRC}")
    parser.add_argument("-d", "--dst", type=str, default=DEFAULT_DST,
                        help=f"directory to save json files generated, defaults to {DEFAULT_DST}")
    parser.add_argument("--force", help="overwrite if file exists", action="store_true")

    arg = parser.parse_args()

    # conf paths
    src = os.path.abspath(os.path.join(os.getcwd(), arg.src))
    source_paths = []
    if not os.path.exists(src):
        raise ValueError(f"{src} not exists")
    if os.path.isfile(src):
        source_paths.append(src)
    elif os.path.isdir(src):
        for name in os.listdir(src):
            file_path = os.path.join(src, name)
            if os.path.isfile(file_path):
                source_paths.append(file_path)
    paths = [(os.path.splitext(os.path.basename(file))[0], file) for file in source_paths if file.endswith(".json")]

    # dst path
    dst_dir = os.path.abspath(os.path.join(os.getcwd(), arg.dst))
    if not os.path.isdir(dst_dir):
        raise ValueError(f"{dst_dir} should be a directory")

    with open(TEMPLATE, "r") as fin:
        temp = fin.read()

    print("[[src files]]")
    print("\n".join([pair[1] for pair in paths]))
    print(f"[[dst directory]]")
    print(dst_dir)
    s = input("continue? (Y/N)\n")
    if s.strip().lower() != "y":
        print("exit")
        sys.exit()

    for file_name, file_path in paths:
        print(f"processing {file_path}")
        with open(file_path, "r") as fin:
            transfer_variable_dict = json.loads(fin.read())
        assert len(transfer_variable_dict) == 1, "multi class defined in a single json"
        class_name, variables = list(transfer_variable_dict.items())[0]

        # generate codes
        variable_define_codes = []
        for pair in variables.items():
            variable_name, role = pair
            src = role['src']
            dst = role['dst']
            if isinstance(src, str):
                src = [src]
            if isinstance(dst, str):
                dst = [dst]
            variable_define_codes.append(
                f"self.{variable_name} = self._create_variable(name='{variable_name}', src={src}, dst={dst})")
        create_variable = f"\n{SPACES}{SPACES}".join(variable_define_codes)
        code_str = temp.format(class_name=class_name, create_variable=create_variable)

        # save to file
        save_path = os.path.join(dst_dir, f"{file_name}_transfer_variable.py")
        if os.path.exists(save_path):
            if not arg.force:
                print(f"file {save_path} exists, do nothing, use --force option to overwrite")
                continue
        with open(save_path, "w") as writer:
            writer.write(code_str)
    print("done")
