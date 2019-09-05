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
import sys

from arch.api.utils import file_utils

BASE_DIR = file_utils.get_project_base_directory()
TRANSFER_VARIABLE_TEMPLATE = os.path.join(BASE_DIR, "federatedml",
                                          "transfer_variable",
                                          "transfer_variable_template.py")


class TransferVariableGenerator(object):
    def __init__(self):
        pass

    @staticmethod
    def write_out_class(writer, class_name, transfer_var_dict, with_header=True):
        if with_header:
            global TRANSFER_VARIABLE_TEMPLATE

            with open(TRANSFER_VARIABLE_TEMPLATE, "r") as fin:
                writer.write(fin.read())

        writer.write("# noinspection PyAttributeOutsideInit\n")
        writer.write("class " + class_name + "(BaseTransferVariable):" + "\n")

        tag = '    '
        writer.write(tag + "def define_transfer_variable(self):" + "\n")

        for transfer_var, auth_dict in transfer_var_dict.items():
            writer.write(tag + tag)
            var_name = class_name + "." + transfer_var
            src_auth = auth_dict['src']
            dst_auth = auth_dict['dst']
            writer.write(
                f"self.{transfer_var} = Variable(name='{var_name}', auth=dict(src='{src_auth}', dst={dst_auth}), transfer_variable=self)")
            writer.write("\n")
        writer.write(tag + tag + "pass\n")
        writer.flush()

    def generate_all(self):
        global BASE_DIR
        conf_dir = os.path.join(BASE_DIR, "federatedml", "transfer_variable", "definition")
        merge_conf_path = os.path.join(conf_dir, "transfer_conf.json")
        trans_var_dir = os.path.join(BASE_DIR, "federatedml", "transfer_variable", "transfer_class")

        merge_dict = {}
        with open(merge_conf_path, "w") as fin:
            pass

        for conf in os.listdir(conf_dir):
            if not conf.endswith(".json"):
                continue

            if conf == "transfer_conf.json":
                continue

            with open(os.path.join(conf_dir, conf), "r") as fin:
                var_dict = json.loads(fin.read())
                merge_dict.update(var_dict)

            out_path = os.path.join(trans_var_dir, conf.split(".", -1)[0] + "_transfer_variable.py")
            fout = open(out_path, "w")
            with_header = True
            for class_name in var_dict:
                transfer_var_dict = var_dict[class_name]
                self.write_out_class(fout, class_name, transfer_var_dict, with_header)
                with_header = False

            fout.flush()
            fout.close()

        with open(merge_conf_path, "w") as fout:
            jsonDumpsIndentStr = json.dumps(merge_dict, indent=1);
            buffers = jsonDumpsIndentStr.split("\n", -1)
            for buf in buffers:
                fout.write(buf + "\n")

    def generate_transfer_var_class(self, transfer_var_conf_path, out_path):
        base_dir = file_utils.get_project_base_directory()
        merge_conf_path = os.path.join(base_dir, "federatedml/transfer_variable_conf/transfer_conf.json")

        merge_dict = {}
        if os.path.isfile(merge_conf_path):
            with open(merge_conf_path, "r") as fin:
                merge_dict = json.loads(fin.read())

        var_dict = {}
        with open(transfer_var_conf_path) as fin:
            var_dict = json.loads(fin.read())

        merge_dict.update(var_dict)

        with open(merge_conf_path, "w") as fout:
            json_dumps_indent_str = json.dumps(merge_dict, indent=1);
            buffers = json_dumps_indent_str.split("\n", -1)
            for buf in buffers:
                fout.write(buf + "\n")

        fout = open(out_path, "w")
        with_header = True
        for class_name in var_dict:
            transfer_var_dict = var_dict[class_name]
            self.write_out_class(fout, class_name, transfer_var_dict, with_header)
            with_header = False

        fout.flush()
        fout.close()


if __name__ == "__main__":

    conf_path = None
    out_path = None

    if len(sys.argv) == 2:
        out_path = sys.argv[1]
    elif len(sys.argv) == 3:
        conf_path = sys.argv[1]
        out_path = sys.argv[2]

    transfer_var_gen = TransferVariableGenerator()
    if conf_path is None and out_path is None:
        transfer_var_gen.generate_all()
    else:
        transfer_var_gen.generate_transfer_var_class(conf_path, out_path)
