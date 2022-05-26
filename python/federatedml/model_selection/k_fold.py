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

import copy
import functools

import numpy as np
from sklearn.model_selection import KFold as sk_KFold

from fate_arch.session import computing_session as session
from federatedml.evaluation.evaluation import Evaluation
from federatedml.model_selection.cross_validate import BaseCrossValidator
from federatedml.model_selection.indices import collect_index
from federatedml.transfer_variable.transfer_class.cross_validation_transfer_variable import \
    CrossValidationTransferVariable
from federatedml.util import LOGGER
from federatedml.util import consts


class KFold(BaseCrossValidator):
    def __init__(self):
        super(KFold, self).__init__()
        self.model_param = None
        self.n_splits = 1
        self.shuffle = True
        self.random_seed = 1
        self.fold_history = None

    def _init_model(self, param):
        self.model_param = param
        self.n_splits = param.n_splits
        self.mode = param.mode
        self.role = param.role
        self.shuffle = param.shuffle
        self.random_seed = param.random_seed
        self.output_fold_history = param.output_fold_history
        self.history_value_type = param.history_value_type
        # self.evaluate_param = param.evaluate_param
        # np.random.seed(self.random_seed)

    def split(self, data_inst):
        # header = data_inst.schema.get('header')
        schema = data_inst.schema

        data_sids_iter, data_size = collect_index(data_inst)
        data_sids = []
        key_type = None
        for sid, _ in data_sids_iter:
            if key_type is None:
                key_type = type(sid)
            data_sids.append(sid)
        data_sids = np.array(data_sids)
        # if self.shuffle:
        #     np.random.shuffle(data_sids)
        random_state = self.random_seed if self.shuffle else None
        kf = sk_KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=random_state)

        n = 0
        for train, test in kf.split(data_sids):
            train_sids = data_sids[train]
            test_sids = data_sids[test]

            n += 1

            train_sids_table = [(key_type(x), 1) for x in train_sids]
            test_sids_table = [(key_type(x), 1) for x in test_sids]
            train_table = session.parallelize(train_sids_table,
                                              include_key=True,
                                              partition=data_inst.partitions)
            train_data = data_inst.join(train_table, lambda x, y: x)

            test_table = session.parallelize(test_sids_table,
                                             include_key=True,
                                             partition=data_inst.partitions)
            test_data = data_inst.join(test_table, lambda x, y: x)
            train_data.schema = schema
            test_data.schema = schema
            yield train_data, test_data

    @staticmethod
    def generate_new_id(id, fold_num, data_type):
        return f"{id}#fold{fold_num}#{data_type}"

    def transform_history_data(self, data, predict_data, fold_num, data_type):
        if self.history_value_type == "score":
            if predict_data is not None:
                history_data = predict_data.map(lambda k, v: (KFold.generate_new_id(k, fold_num, data_type), v))
                history_data.schema = copy.deepcopy(predict_data.schema)
            else:
                history_data = data.map(lambda k, v: (KFold.generate_new_id(k, fold_num, data_type), fold_num))
                schema = copy.deepcopy(data.schema)
                schema["header"] = ["fold_num"]
                history_data.schema = schema

        elif self.history_value_type == "instance":
            history_data = data.map(lambda k, v: (KFold.generate_new_id(k, fold_num, data_type), v))
            history_data.schema = copy.deepcopy(data.schema)
        else:
            raise ValueError(f"unknown history value type")
        return history_data

    @staticmethod
    def _append_name(instance, name):
        new_inst = copy.deepcopy(instance)
        new_inst.features.append(name)
        return new_inst

    def run(self, component_parameters, data_inst, original_model, host_do_evaluate):
        self._init_model(component_parameters)

        if data_inst is None:
            self._arbiter_run(original_model)
            return
        total_data_count = data_inst.count()
        LOGGER.debug(f"data_inst count: {total_data_count}")
        if self.output_fold_history:
            if total_data_count * self.n_splits > consts.MAX_SAMPLE_OUTPUT_LIMIT:
                LOGGER.warning(
                    f"max sample output limit {consts.MAX_SAMPLE_OUTPUT_LIMIT} exceeded with n_splits ({self.n_splits}) * instance_count ({total_data_count})")
        if self.mode == consts.HOMO or self.role == consts.GUEST:
            data_generator = self.split(data_inst)
        else:
            data_generator = [(data_inst, data_inst)] * self.n_splits
        fold_num = 0

        summary_res = {}
        for train_data, test_data in data_generator:
            model = copy.deepcopy(original_model)
            LOGGER.debug("In CV, set_flowid flowid is : {}".format(fold_num))
            model.set_flowid(fold_num)
            model.set_cv_fold(fold_num)

            LOGGER.info("KFold fold_num is: {}".format(fold_num))
            if self.mode == consts.HETERO:
                train_data = self._align_data_index(train_data, model.flowid, consts.TRAIN_DATA)
                LOGGER.info("Train data Synchronized")
                test_data = self._align_data_index(test_data, model.flowid, consts.TEST_DATA)
                LOGGER.info("Test data Synchronized")
            train_data_count = train_data.count()
            test_data_count = test_data.count()
            LOGGER.debug(f"train_data count: {train_data_count}")
            if train_data_count + test_data_count != total_data_count:
                raise EnvironmentError("In cv fold: {}, train count: {}, test count: {}, original data count: {}."
                                       "Thus, 'train count + test count = total count' condition is not satisfied"
                                       .format(fold_num, train_data_count, test_data_count, total_data_count))
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
                f = functools.partial(self._append_name, name='train')
                train_pred_res = train_pred_res.mapValues(f)
                train_pred_res = model.set_predict_data_schema(train_pred_res, train_data.schema)
                # LOGGER.debug(f"train_pred_res schema: {train_pred_res.schema}")
                self.evaluate(train_pred_res, fold_name, model)

            this_flowid = 'predict_validate.' + str(fold_num)
            LOGGER.debug("In CV, set_flowid flowid is : {}".format(this_flowid))
            model.set_flowid(this_flowid)
            test_pred_res = model.predict(test_data)

            # if pred_res is not None:
            if self.role == consts.GUEST or host_do_evaluate:
                fold_name = "_".join(['validate', 'fold', str(fold_num)])
                f = functools.partial(self._append_name, name='validate')
                test_pred_res = test_pred_res.mapValues(f)
                test_pred_res = model.set_predict_data_schema(test_pred_res, test_data.schema)
                # LOGGER.debug(f"train_pred_res schema: {test_pred_res.schema}")
                self.evaluate(test_pred_res, fold_name, model)
            LOGGER.debug("Finish fold: {}".format(fold_num))

            if self.output_fold_history:
                LOGGER.debug(f"generating fold history for fold {fold_num}")
                fold_train_data = self.transform_history_data(train_data, train_pred_res, fold_num, "train")
                fold_validate_data = self.transform_history_data(test_data, test_pred_res, fold_num, "validate")

                fold_history_data = fold_train_data.union(fold_validate_data)
                fold_history_data.schema = fold_train_data.schema
                if self.fold_history is None:
                    self.fold_history = fold_history_data
                else:
                    new_fold_history = self.fold_history.union(fold_history_data)
                    new_fold_history.schema = fold_history_data.schema
                    self.fold_history = new_fold_history

            summary_res[f"fold_{fold_num}"] = model.summary()
            fold_num += 1
        summary_res['fold_num'] = fold_num
        LOGGER.debug("Finish all fold running")
        original_model.set_summary(summary_res)
        if self.output_fold_history:
            LOGGER.debug(f"output data schema: {self.fold_history.schema}")
            # LOGGER.debug(f"output data: {list(self.fold_history.collect())}")
            # LOGGER.debug(f"output data is: {self.fold_history}")
            return self.fold_history
        else:
            return data_inst

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
        schema = data_instance.schema

        if data_application is None:
            # LOGGER.warning("not data_application!")
            # return
            raise ValueError("In _align_data_index, data_application should be provided.")

        transfer_variable = CrossValidationTransferVariable()
        if data_application == consts.TRAIN_DATA:
            transfer_id = transfer_variable.train_sid
        elif data_application == consts.TEST_DATA:
            transfer_id = transfer_variable.test_sid
        else:
            raise ValueError("In _align_data_index, data_application should be provided.")

        if self.role == consts.GUEST:
            data_sid = data_instance.mapValues(lambda v: 1)
            transfer_id.remote(data_sid,
                               role=consts.HOST,
                               idx=-1,
                               suffix=(flowid,))
            LOGGER.info("remote {} to host".format(data_application))
            return data_instance
        elif self.role == consts.HOST:
            data_sid = transfer_id.get(idx=0,
                                       suffix=(flowid,))

            LOGGER.info("get {} from guest".format(data_application))
            join_data_insts = data_sid.join(data_instance, lambda s, d: d)
            join_data_insts.schema = schema
            return join_data_insts

    def evaluate(self, validate_data, fold_name, model):

        if validate_data is None:
            return

        eval_obj = Evaluation()
        # LOGGER.debug("In KFold, evaluate_param is: {}".format(self.evaluate_param.__dict__))
        # eval_obj._init_model(self.evaluate_param)
        eval_param = model.get_metrics_param()

        eval_param.check_single_value_default_metric()
        eval_obj._init_model(eval_param)
        eval_obj.set_tracker(model.tracker)
        validate_data = {fold_name: validate_data}
        eval_obj.fit(validate_data)
        eval_obj.save_data()
