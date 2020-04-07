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
from arch.api import session
from arch.api.utils import log_utils
from federatedml.model_selection.cross_validate import BaseCrossValidator
from federatedml.model_selection.indices import collect_index
from federatedml.util import consts
from federatedml.evaluation.evaluation import Evaluation
from federatedml.transfer_variable.transfer_class.cross_validation_transfer_variable import CrossValidationTransferVariable

LOGGER = log_utils.getLogger()


class KFold(BaseCrossValidator):
    def __init__(self):
        super(KFold, self).__init__()
        self.model_param = None
        self.n_splits = 1
        self.shuffle = True
        self.random_seed = 1

    def _init_model(self, param):
        self.model_param = param
        self.n_splits = param.n_splits
        self.mode = param.mode
        self.role = param.role
        self.shuffle = param.shuffle
        self.random_seed = param.random_seed
        # self.evaluate_param = param.evaluate_param
        np.random.seed(self.random_seed)

    def split(self, data_inst):
        np.random.seed(self.random_seed)

        header = data_inst.schema.get('header')

        data_sids_iter, data_size = collect_index(data_inst)
        data_sids = []
        key_type = None
        for sid, _ in data_sids_iter:
            if key_type is None:
                key_type = type(sid)
            data_sids.append(sid)
        data_sids = np.array(data_sids)
        if self.shuffle:
            np.random.shuffle(data_sids)

        kf = sk_KFold(n_splits=self.n_splits)

        n = 0
        for train, test in kf.split(data_sids):

            train_sids = data_sids[train]
            test_sids = data_sids[test]

            n += 1

            train_sids_table = [(key_type(x), 1) for x in train_sids]
            test_sids_table = [(key_type(x), 1) for x in test_sids]
            # print(train_sids_table)
            train_table = session.parallelize(train_sids_table,
                                              include_key=True,
                                              partition=data_inst._partitions)
            train_data = data_inst.join(train_table, lambda x, y: x)

            test_table = session.parallelize(test_sids_table,
                                             include_key=True,
                                             partition=data_inst._partitions)
            test_data = data_inst.join(test_table, lambda x, y: x)
            train_data.schema['header'] = header
            test_data.schema['header'] = header
            yield train_data, test_data

    def run(self, component_parameters, data_inst, original_model, host_do_evaluate):
        self._init_model(component_parameters)

        if data_inst is None:
            self._arbiter_run(original_model)
            return

        LOGGER.debug("data_inst count: {}".format(data_inst.count()))
        data_generator = self.split(data_inst)
        fold_num = 0
        for train_data, test_data in data_generator:
            model = copy.deepcopy(original_model)
            LOGGER.debug("In CV, set_flowid flowid is : {}".format(fold_num))
            model.set_flowid(fold_num)
            model.set_cv_fold(fold_num)

            LOGGER.info("KFold fold_num is: {}".format(fold_num))
            if self.mode == consts.HETERO:
                self._align_data_index(train_data, model.flowid, consts.TRAIN_DATA)
                LOGGER.info("Train data Synchronized")
                self._align_data_index(test_data, model.flowid, consts.TEST_DATA)
                LOGGER.info("Test data Synchronized")
            LOGGER.debug("train_data count: {}".format(train_data.count()))

            this_flowid = 'train.' + str(fold_num)
            LOGGER.debug("In CV, set_flowid flowid is : {}".format(this_flowid))
            model.set_flowid(this_flowid)
            model.fit(train_data, test_data)

            this_flowid = 'predict_train.' + str(fold_num)
            LOGGER.debug("In CV, set_flowid flowid is : {}".format(this_flowid))
            model.set_flowid(this_flowid)
            train_pred_res = model.predict(train_data)

            # if train_pred_res is not None:
            if self.role == consts.GUEST or host_do_evaluate:
                fold_name = "_".join(['train', 'fold', str(fold_num)])
                pred_res = train_pred_res.mapValues(lambda value: value + ['train'])
                self.evaluate(pred_res, fold_name, model)

            this_flowid = 'predict_validate.' + str(fold_num)
            LOGGER.debug("In CV, set_flowid flowid is : {}".format(this_flowid))
            model.set_flowid(this_flowid)
            pred_res = model.predict(test_data)
            model.set_predict_data_schema(pred_res, test_data.schema)

            # if pred_res is not None:
            if self.role == consts.GUEST or host_do_evaluate:
                fold_name = "_".join(['validate', 'fold', str(fold_num)])
                pred_res = pred_res.mapValues(lambda value: value + ['validate'])
                self.evaluate(pred_res, fold_name, model)
            fold_num += 1
            LOGGER.debug("Finish fold: {}".format(fold_num))
        LOGGER.debug("Finish all fold running")

        return

    def _arbiter_run(self, original_model):
        for fold_num in range(self.n_splits):
            LOGGER.info("KFold flowid is: {}".format(fold_num))
            model = copy.deepcopy(original_model)
            this_flowid = 'train.' + str(fold_num)
            model.set_flowid(this_flowid)
            model.set_cv_fold(fold_num)
            model.fit(None)

            this_flowid = 'predict_train.' + str(fold_num)
            model.set_flowid(this_flowid)
            model.predict(None)

            this_flowid = 'predict_validate.' + str(fold_num)
            model.set_flowid(this_flowid)
            model.predict(None)

    def _align_data_index(self, data_instance, flowid, data_application=None):
        header = data_instance.schema.get('header')

        if data_application is None:
            LOGGER.warning("not data_application!")
            return

        transfer_variable = CrossValidationTransferVariable()
        if data_application == consts.TRAIN_DATA:
            transfer_id = transfer_variable.train_sid
        elif data_application == consts.TEST_DATA:
            transfer_id = transfer_variable.test_sid
        else:
            LOGGER.warning("data_application error!")
            return

        if self.role == consts.GUEST:
            data_sid = data_instance.mapValues(lambda v: 1)
            transfer_id.remote(data_sid,
                               role=consts.HOST,
                               idx=-1,
                               suffix=(flowid,))
            LOGGER.info("remote {} to host".format(data_application))
            return None
        elif self.role == consts.HOST:
            data_sid = transfer_id.get(idx=0,
                                       suffix=(flowid,))

            LOGGER.info("get {} from guest".format(data_application))
            join_data_insts = data_sid.join(data_instance, lambda s, d: d)
            join_data_insts.schema['header'] = header
            return join_data_insts

    def evaluate(self, eval_data, fold_name, model):
        if eval_data is None:
            return
        eval_obj = Evaluation()
        # LOGGER.debug("In KFold, evaluate_param is: {}".format(self.evaluate_param.__dict__))
        # eval_obj._init_model(self.evaluate_param)
        eval_param = model.get_metrics_param()
        eval_obj._init_model(eval_param)
        eval_obj.set_tracker(model.tracker)
        eval_data = {fold_name: eval_data}
        eval_obj.fit(eval_data)
        eval_obj.save_data()

