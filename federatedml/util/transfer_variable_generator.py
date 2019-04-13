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
################################################################################
#
#
################################################################################

# =============================================================================
# Transfer Variable Generator.py
# =============================================================================

import json
import os
import sys
from arch.api.utils import file_utils

header = ["#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n",
          "################################################################################\n",
          "#\n# Copyright (c) 2018, WeBank Inc. All Rights Reserved\n#\n",
          "################################################################################\n",
          "# =============================================================================\n",
          "# TransferVariable Class\n# =============================================================================\n"]

base_class = 'class Variable(object):\n    def __init__(self, name, auth):\n        ' + \
             'self.name = name\n        self.auth = auth\n\nclass BaseTransferVariable(object):\n    ' + \
             'def __init__(self, flowid=0):\n        self.flowid = flowid\n        ' + \
             'self.define_transfer_variable()\n\n    def set_flowid(self, flowid):\n        ' + \
             'self.flowid = flowid\n\n    def generate_transferid(self, transfer_var, *suffix):\n        ' + \
             'if transfer_var.name.split(".", -1)[-1] not in self.__dict__:\n            ' + \
             'raise ValueError("transfer variable not in class, please check if!!!")\n\n        ' + \
             'transferid = transfer_var.name + "." + str(self.flowid)\n        if suffix:\n            ' + \
             'transferid += "." + ".".join(map(str, suffix))\n        return transferid\n\n' + \
             '    def define_transfer_variable(self):\n        pass\n'


class TransferVariableGenerator(object):
    def __init__(self, conf_path=None, out_path=None):
        self.conf = conf_path
        self.out_path = out_path

    def write_base_class(self, writer):
        writer.write(base_class)

    def write_out_class(self, writer, class_name, transfer_var_dict):
        writer.write("class " + class_name + "(BaseTransferVariable):" + "\n")

        tag = '    '
        writer.write(tag + "def define_transfer_variable(self):" + "\n")

        for transfer_var, auth_dict in transfer_var_dict.items():
            writer.write(tag + tag)
            var_name = class_name + "." + transfer_var
            writer.write("self." + transfer_var + " = ")
            writer.write("Variable(name=" + '"' + var_name + '"' + ", ")
            writer.write("auth=" + "{'src': " + '"' + auth_dict["src"] + '"' + ", " + \
                         "'dst': " + str(auth_dict["dst"]) + "})")

            writer.write("\n")

        writer.write(tag + tag + "pass\n")
        writer.flush()

    def run(self):
        if self.conf is not None and self.out_path is not None:
            with open(self.conf, "r") as fin:
                buf = fin.read()
                conf_dict = json.loads(buf)

            fout = open(self.out_path, "w")
            for head in header:
                fout.write(head.strip() + "\n")

            fout.write("\n")
            fout.write("\n")

            self.write_base_class(fout)

            for class_name in conf_dict:
                transfer_var_dict = conf_dict[class_name]
                fout.write("\n\n")
                self.write_out_class(fout, class_name, transfer_var_dict)
            fout.flush()
            fout.close()
        else:
            base_dir = file_utils.get_project_base_directory()
            conf_dir = os.path.join(base_dir, "federatedml/transfer_variable_conf/")
            out_file = os.path.join(base_dir, "federatedml/util/transfer_variable.py")
            merge_conf = os.path.join(conf_dir, "transfer_conf.json")
           
            merge_dict = {}

            with open(out_file, "w") as fout:
                for head in header:
                    fout.write(head.strip() + "\n")
                
                fout.write("\n")
                fout.write("\n")
                
                self.write_base_class(fout)
                
                for json_conf in os.listdir(conf_dir):
                    if not json_conf.endswith(".json"):
                        continue

                    with open(os.path.join(conf_dir, json_conf), "r") as fin:
                        buf = fin.read()
                        conf_dict = json.loads(buf)
            

                    for class_name in conf_dict:
                        if class_name in merge_dict:
                            continue

                        transfer_var_dict = conf_dict[class_name]
                        fout.write("\n\n")
                        self.write_out_class(fout, class_name, transfer_var_dict)
           
                    merge_dict.update(conf_dict)

                fout.flush()

            with open(merge_conf, "w") as fout:
                fout.write(json.dumps(merge_dict) + "\n")


if __name__ == "__main__":
    config_path = None
    out_path = None
    if len(sys.argv) > 2:
        config_path = sys.argv[1]
        out_path = sys.argv[2]

    transfervar_gen = TransferVariableGenerator(config_path, out_path)
    transfervar_gen.run()
