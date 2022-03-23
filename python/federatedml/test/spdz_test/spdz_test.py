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

import numpy as np
import time
from prettytable import PrettyTable, ORGMODE
from fate_arch.session import computing_session as session, get_parties
from federatedml.secureprotol.spdz import SPDZ
from federatedml.model_base import ModelBase, ComponentOutput
from federatedml.test.spdz_test.spdz_test_param import SPDZTestParam
from federatedml.util import LOGGER
from federatedml.secureprotol.spdz.tensor.fixedpoint_table import FixedPointTensor as TableTensor
from federatedml.secureprotol.spdz.tensor.fixedpoint_numpy import FixedPointTensor as NumpyTensor


class SPDZTest(ModelBase):
    def __init__(self):
        super(SPDZTest, self).__init__()
        self.data_num = None
        self.data_partition = None
        self.seed = None
        self.test_round = None

        self.tracker = None

        """plaintest data"""
        self.int_data_x = None
        self.float_data_x = None
        self.int_data_y = None
        self.float_data_y = None

        self.model_param = SPDZTestParam()
        self.parties = None
        self.local_party = None
        self.other_party = None
        self._set_parties()
        self.metric = None
        self.operation = None
        self.test_count = None
        self.op_test_list = ["float_add", "int_add", "float_sub", "int_sub", "float_dot", "int_dot"]

        self._summary = {"op_test_list": self.op_test_list,
                         "tensor_type": ["numpy", "table"],
                         "numpy": {},
                         "table": {}}

    def _init_runtime_parameters(self, cpn_input):
        self.model_param.update(cpn_input.parameters)
        self.tracker = cpn_input.tracker
        self._init_model()

    def _init_model(self):
        self.data_num = self.model_param.data_num
        self.data_partition = self.model_param.data_partition
        self.seed = self.model_param.seed
        self.test_round = self.model_param.test_round
        self.data_lower_bound = self.model_param.data_lower_bound
        self.data_upper_bound = self.model_param.data_upper_bound

        self.data_lower_bound = 0
        self.data_upper_bound = 100

    def _set_parties(self):
        parties = []
        guest_parties = get_parties().roles_to_parties(["guest"])
        host_parties = get_parties().roles_to_parties(["host"])
        parties.extend(guest_parties)
        parties.extend(host_parties)

        local_party = get_parties().local_party
        other_party = parties[0] if parties[0] != local_party else parties[1]

        self.parties = parties
        self.local_party = local_party
        self.other_party = other_party

    def _init_data(self):
        np.random.seed(self.seed)
        self.int_data_x = np.random.randint(int(self.data_lower_bound), int(self.data_upper_bound), size=self.data_num)
        self.float_data_x = np.random.uniform(self.data_lower_bound, self.data_upper_bound, size=self.data_num)
        self.int_data_y = np.random.randint(int(self.data_lower_bound), int(self.data_upper_bound), size=self.data_num)
        self.float_data_y = np.random.uniform(self.data_lower_bound, self.data_upper_bound, size=self.data_num)

    def _test_spdz(self):
        table_list = []
        table_int_data_x, table_float_data_x = None, None
        table_int_data_y, table_float_data_y = None, None
        if self.local_party.role == "guest":
            table_int_data_x = session.parallelize(self.int_data_x,
                                                   include_key=False,
                                                   partition=self.data_partition)
            table_int_data_x = table_int_data_x.mapValues(lambda x: np.array([x]))
            table_float_data_x = session.parallelize(self.float_data_x,
                                                     include_key=False,
                                                     partition=self.data_partition)
            table_float_data_x = table_float_data_x.mapValues(lambda x: np.array([x]))
        else:
            table_int_data_y = session.parallelize(self.int_data_y,
                                                   include_key=False,
                                                   partition=self.data_partition)
            table_int_data_y = table_int_data_y.mapValues(lambda y: np.array([y]))
            table_float_data_y = session.parallelize(self.float_data_y,
                                                     include_key=False,
                                                     partition=self.data_partition)
            table_float_data_y = table_float_data_y.mapValues(lambda y: np.array([y]))

        for tensor_type in ["numpy", "table"]:
            table = PrettyTable()
            table.set_style(ORGMODE)
            field_name = ["DataType", "One time consumption", f"{self.data_num} times consumption",
                          "relative acc", "log2 acc", "operations per second"]
            self._summary["field_name"] = field_name
            table.field_names = field_name

            with SPDZ(local_party=self.local_party, all_parties=self.parties) as spdz:
                for op_type in self.op_test_list:
                    start_time = time.time()
                    for epoch in range(self.test_round):
                        LOGGER.info(f"test spdz, tensor_type: {tensor_type}, op_type: {op_type}, epoch: {epoch}")
                        tag = "_".join([tensor_type, op_type, str(epoch)])
                        spdz.set_flowid(tag)
                        if self.local_party.role == "guest":
                            if tensor_type == "table":
                                if op_type.startswith("int"):
                                    fixed_point_x = TableTensor.from_source("int_x_" + tag, table_int_data_x)
                                    fixed_point_y = TableTensor.from_source("int_y_" + tag, self.other_party)
                                else:
                                    fixed_point_x = TableTensor.from_source("float_x_" + tag, table_float_data_x)
                                    fixed_point_y = TableTensor.from_source("float_y_" + tag, self.other_party)
                            else:
                                if op_type.startswith("int"):
                                    fixed_point_x = NumpyTensor.from_source("int_x_" + tag, self.int_data_x)
                                    fixed_point_y = NumpyTensor.from_source("int_y_" + tag, self.other_party)
                                else:
                                    fixed_point_x = NumpyTensor.from_source("float_x_" + tag, self.float_data_x)
                                    fixed_point_y = NumpyTensor.from_source("float_y_" + tag, self.other_party)
                        else:
                            if tensor_type == "table":
                                if op_type.startswith("int"):
                                    fixed_point_y = TableTensor.from_source("int_y_" + tag, table_int_data_y)
                                    fixed_point_x = TableTensor.from_source("int_x_" + tag, self.other_party)
                                else:
                                    fixed_point_y = TableTensor.from_source("float_y_" + tag, table_float_data_y)
                                    fixed_point_x = TableTensor.from_source("float_x_" + tag, self.other_party)
                            else:
                                if op_type.startswith("int"):
                                    fixed_point_y = NumpyTensor.from_source("int_y_" + tag, self.int_data_y)
                                    fixed_point_x = NumpyTensor.from_source("int_x_" + tag, self.other_party)
                                else:
                                    fixed_point_y = NumpyTensor.from_source("float_y_" + tag, self.float_data_y)
                                    fixed_point_x = NumpyTensor.from_source("float_x_" + tag, self.other_party)

                        ret = self.calculate_ret(op_type, tensor_type, fixed_point_x, fixed_point_y)

                    total_time = time.time() - start_time
                    self.output_table(op_type, table, tensor_type, total_time, ret)

            table_list.append(table)

        self.tracker.log_component_summary(self._summary)
        for table in table_list:
            LOGGER.info(table)

    def calculate_ret(self, op_type, tensor_type,
                      fixed_point_x, fixed_point_y,
                      ):
        if op_type.endswith("add"):
            ret = (fixed_point_x + fixed_point_y).get()
        elif op_type.endswith("sub"):
            ret = (fixed_point_x - fixed_point_y).get()
        else:
            ret = (fixed_point_x.dot(fixed_point_y)).get()[0]
            if tensor_type == "table":
                ret = ret[0]

        if tensor_type == "table" and not op_type.endswith("dot"):
            arr = [None] * self.data_num
            for k, v in ret.collect():
                arr[k] = v[0]
            ret = np.array(arr)

        return ret

    def output_table(self, op_type, table, tensor_type, total_time, spdz_ret):
        if op_type.startswith("int"):
            data_x = self.int_data_x
            data_y = self.int_data_y
        else:
            data_x = self.float_data_x
            data_y = self.float_data_y

        numpy_ret = None
        if op_type.endswith("add") or op_type.endswith("sub"):
            start = time.time()
            for i in range(self.test_round):
                if op_type.endswith("add"):
                    numpy_ret = data_x + data_y
                else:
                    numpy_ret = data_x - data_y
            plain_text_time = time.time() - start
            relative_acc = 0
            for np_x, spdz_x in zip(numpy_ret, spdz_ret):
                relative_acc += abs(np_x - spdz_x) / max(abs(np_x), abs(spdz_x) + 1e-15)
        else:
            start = time.time()
            for i in range(self.test_round):
                numpy_ret = np.dot(data_x, data_y)
            plain_text_time = time.time() - start
            relative_acc = abs(numpy_ret - spdz_ret) / max(abs(numpy_ret), abs(spdz_ret))

        relative_acc /= self.data_num
        log2_acc = -np.log2(relative_acc) if relative_acc != 0 else 0

        row_info = [op_type, total_time / self.data_num / self.test_round, total_time / self.test_round,
                    relative_acc, log2_acc, int(self.data_num * self.test_round / total_time)]
        table.add_row(row_info)
        self._summary[tensor_type][op_type] = row_info

        return table.get_string(title=f"SPDZ {tensor_type} Computational performance")

    def run(self, cpn_input):
        LOGGER.info("begin to init parameters of secure add example")

        self._init_runtime_parameters(cpn_input)
        LOGGER.info("begin to make data")
        self._init_data()

        self._test_spdz()

        return ComponentOutput(self.save_data(), self.export_model(), self.save_cache())
