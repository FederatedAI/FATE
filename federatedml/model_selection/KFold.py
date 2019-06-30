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
from sklearn.model_selection import KFold as sk_KFold
import copy
from arch.api import eggroll
from arch.api import federation
from arch.api.utils import log_utils
from federatedml.model_selection.cross_validate import BaseCrossValidator
from federatedml.model_selection.indices import collect_index
from federatedml.util import consts
from federatedml.util.transfer_variable.hetero_workflow_transfer_variable import HeteroWorkFlowTransferVariable

LOGGER = log_utils.getLogger()


class KFold(BaseCrossValidator):
    def __init__(self):
        super(KFold, self).__init__()
        self.model_param = None
        self.n_splits = 1
        self.shuffle = True
        self.random_seed = 1

    def split(self, data_inst):
        header = data_inst.schema.get('header')

        data_sids_iter, data_size = collect_index(data_inst)

        data_sids = []
        for sid, _ in data_sids_iter:
            data_sids.append(sid)
        data_sids = np.array(data_sids)

        if self.shuffle:
            np.random.shuffle(data_sids)

        kf = sk_KFold(n_splits=self.n_splits)

        for train, test in kf.split(data_sids):
            train_sids = data_sids[train]
            test_sids = data_sids[test]
            train_sids_table = [(str(x), 1) for x in train_sids]
            test_sids_table = [(str(x), 1) for x in test_sids]
            # print(train_sids_table)
            train_table = eggroll.parallelize(train_sids_table,
                                              include_key=True,
                                              partition=data_inst._partitions)
            train_data = data_inst.join(train_table, lambda x, y: x)
            test_table = eggroll.parallelize(test_sids_table,
                                             include_key=True,
                                             partition=data_inst._partitions)
            test_data = data_inst.join(test_table, lambda x, y: x)
            train_data.schema['header'] = header
            test_data.schema['header'] = header
            yield train_data, test_data

    def run(self, component_parameters, data_inst, original_model):
        self._init_model(component_parameters)

        if data_inst is None:
            cv_results = self._arbiter_run(original_model)
            return cv_results

        data_generator = self.split(data_inst)
        cv_results = []
        flowid = 0
        for train_data, test_data in data_generator:
            LOGGER.info("KFold flowid is: {}".format(flowid))
            if self.mode == consts.HETERO:
                self._synchronize_data(train_data, flowid, consts.TRAIN_DATA)
                LOGGER.info("Train data Synchronized")
                self._synchronize_data(test_data, flowid, consts.TEST_DATA)
                LOGGER.info("Test data Synchronized")
            model = copy.deepcopy(original_model)
            model.set_flowid(flowid)
            model.fit(train_data)
            pred_res = model.predict(test_data)
            evaluation_results = self.evaluate(pred_res, model)

            cv_results.append(evaluation_results)
            flowid += 1
        self.display_cv_result(cv_results)
        return cv_results

    def _arbiter_run(self, original_model):
        cv_results = []
        for flowid in range(self.n_splits):
            LOGGER.info("KFold flowid is: {}".format(flowid))
            model = copy.deepcopy(original_model)
            model.set_flowid(flowid)
            model.fit()
            pred_res = model.predict()
            evaluation_results = self.evaluate(pred_res, model)
            cv_results.append(evaluation_results)
        return cv_results



    def _init_model(self, param):
        self.model_param = param
        self.n_splits = param.n_splits
        self.mode = param.mode
        self.role = param.role
        self.shuffle = param.shuffle
        self.random_seed = param.random_seed
        self.evaluate_param = param.evaluate_param
        np.random.seed(self.random_seed)

    def _synchronize_data(self, data_instance, flowid, data_application=None):
        header = data_instance.schema.get('header')

        if data_application is None:
            LOGGER.warning("not data_application!")
            return

        transfer_variable = HeteroWorkFlowTransferVariable()
        if data_application == consts.TRAIN_DATA:
            transfer_id = transfer_variable.train_data
        elif data_application == consts.TEST_DATA:
            transfer_id = transfer_variable.test_data
        else:
            LOGGER.warning("data_application error!")
            return

        if self.role == consts.GUEST:
            data_sid = data_instance.mapValues(lambda v: 1)

            federation.remote(data_sid,
                              name=transfer_id.name,
                              tag=transfer_variable.generate_transferid(transfer_id, flowid),
                              role=consts.HOST,
                              idx=0)
            LOGGER.info("remote {} to host".format(data_application))
            return None
        elif self.role == consts.HOST:
            data_sid = federation.get(name=transfer_id.name,
                                      tag=transfer_variable.generate_transferid(transfer_id, flowid),
                                      idx=0)

            LOGGER.info("get {} from guest".format(data_application))
            join_data_insts = data_sid.join(data_instance, lambda s, d: d)
            join_data_insts.schema['header'] = header
            return join_data_insts

    def evaluate(self, eval_data, model):
        if eval_data is None:
            return None

        eval_data_local = eval_data.collect()
        labels = []
        pred_prob = []
        pred_labels = []
        data_num = 0
        for data in eval_data_local:
            data_num += 1
            labels.append(data[1][0])
            pred_prob.append(data[1][1])
            pred_labels.append(data[1][2])

        labels = np.array(labels)
        pred_prob = np.array(pred_prob)
        pred_labels = np.array(pred_labels)

        evaluation_result = model.evaluate(labels, pred_prob, pred_labels,
                                           evaluate_param=self.evaluate_param)
        return evaluation_result
