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

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class RunningFuncs(object):
    def __init__(self):
        self.todo_func_list = []
        self.todo_func_params = []
        self.save_result = []
        self.use_previews_result = []

    def add_func(self, func, params, save_result=False, use_previews=False):
        self.todo_func_list.append(func)
        self.todo_func_params.append(params)
        self.save_result.append(save_result)
        self.use_previews_result.append(use_previews)

    def __iter__(self):
        for func, params, save_result, use_previews in zip(self.todo_func_list, self.todo_func_params,
                                                           self.save_result, self.use_previews_result):
            yield func, params, save_result, use_previews


class DSLConfigError(ValueError):
    pass


class ComponentProperties(object):
    def __init__(self):
        self.need_cv = False
        self.need_run = False
        self.need_stepwise = False
        self.has_model = False
        self.has_isometric_model = False
        self.has_train_data = False
        self.has_eval_data = False
        self.has_validate_data = False
        self.has_test_data = False
        self.has_normal_input_data = False
        self.role = None
        self.host_party_idlist = []
        self.local_partyid = -1
        self.guest_partyid = -1
        self.input_data_count = 0
        self.input_eval_data_count = 0

    def parse_component_param(self, component_parameters, param):

        try:
            need_cv = param.cv_param.need_cv
        except AttributeError:
            need_cv = False
        self.need_cv = need_cv
        LOGGER.debug(component_parameters)

        try:
            need_run = param.need_run
        except AttributeError:
            need_run = True
        self.need_run = need_run
        LOGGER.debug("need_run: {}, need_cv: {}".format(self.need_run, self.need_cv))

        try:
            need_stepwise = param.stepwise_param.need_stepwise
        except AttributeError:
            need_stepwise = False
        self.need_stepwise = need_stepwise

        self.role = component_parameters["local"]["role"]
        self.host_party_idlist = component_parameters["role"].get("host")
        self.local_partyid = component_parameters["local"].get("party_id")
        self.guest_partyid = component_parameters["role"].get("guest")
        if self.guest_partyid is not None:
            self.guest_partyid = self.guest_partyid[0]
        return self

    def parse_dsl_args(self, args):
        if "model" in args:
            self.has_model = True
        if "isometric_model" in args:
            self.has_isometric_model = True
        data_sets = args.get("data")
        if data_sets is None:
            return self
        for data_key, data_dicts in data_sets.items():
            data_keys = list(data_dicts.keys())

            for data_type in ["train_data", "eval_data", "validate_data", "test_data"]:
                if data_type in data_keys:
                    setattr(self, f"has_{data_type}", True)
                    data_keys.remove(data_type)
                LOGGER.debug(f"[Data Parser], has_{data_type}:"
                             f" {getattr(self, f'has_{data_type}')}")

            if len(data_keys) > 0:
                self.has_normal_input_data = True

        LOGGER.debug("[Data Parser], has_normal_data: {}".format(self.has_normal_input_data))
        if self.has_eval_data:
            if self.has_validate_data or self.has_test_data:
                raise DSLConfigError("eval_data input should not be configured simultaneously"
                                     " with validate_data or test_data")
        # self._abnormal_dsl_config_detect()
        return self

    def _abnormal_dsl_config_detect(self):
        if self.has_validate_data:
            if not self.has_train_data:
                raise DSLConfigError("validate_data should be configured simultaneously"
                                     " with train_data")

        if self.has_train_data:
            if self.has_normal_input_data or self.has_test_data:
                raise DSLConfigError("train_data input should not be configured simultaneously"
                                     " with data or test_data")

        if self.has_normal_input_data:
            if self.has_train_data or self.has_validate_data or self.has_test_data:
                raise DSLConfigError("When data input has been configured, train_data, "
                                     "validate_data or test_data should not be configured.")

        if self.has_test_data:
            if not self.has_model:
                raise DSLConfigError("When test_data input has been configured, model "
                                     "input should be configured too.")

        if self.has_model:
            if self.has_train_data:
                raise DSLConfigError("train_data input and model input should not be "
                                     "configured simultaneously")
            if self.has_isometric_model:
                raise DSLConfigError("model and isometric_model should not be "
                                     "configured simultaneously")
            if not self.has_test_data and not self.has_normal_input_data:
                raise DSLConfigError("When model has been set, either test_data or "
                                     "data should be provided")

        if self.need_cv or self.need_stepwise:
            if not self.has_train_data:
                raise DSLConfigError("Train_data should be configured in cross-validate "
                                     "task or stepwise task")
            if self.has_validate_data or self.has_normal_input_data or \
                    self.has_test_data:
                raise DSLConfigError("Train_data should be set only in cross-validate "
                                     "task or stepwise task")

            if self.has_model or self.has_isometric_model:
                raise DSLConfigError("In cross-validate task or stepwise task, model "
                                     "or isometric_model should not be configured")

    def extract_input_data(self, args, model):
        data_sets = args.get("data")
        model_data = {}
        data = {}

        if data_sets is None:
            return model_data, data

        LOGGER.debug(f"Input data_sets: {data_sets}")

        for cpn_name, data_dict in data_sets.items():
            for data_type in ["train_data", "eval_data", "validate_data", "test_data"]:
                if data_type in data_dict:
                    d_table = data_dict.get(data_type)
                    model_data[data_type] = model.obtain_data(d_table)
                    del data_dict[data_type]

            if len(data_dict) > 0:
                for k, v in data_dict.items():
                    data_list = model.obtain_data(v)
                    if isinstance(data_list, list):
                        for i, data_i in enumerate(data_list):
                            data[".".join([cpn_name, k, str(i)])] = data_i
                    else:
                        data[".".join([cpn_name, k])] = data_list

        train_data = model_data.get('train_data')
        validate_data = None
        if self.has_train_data:
            if self.has_eval_data:
                validate_data = model_data.get('eval_data')
            elif self.has_validate_data:
                validate_data = model_data.get('validate_data')
        test_data = None
        if self.has_test_data:
            test_data = model_data.get('test_data')
            self.has_test_data = True
        elif self.has_eval_data and not self.has_train_data:
            test_data = model_data.get('eval_data')
            self.has_test_data = True

        # self.has_train_data = True if train_data else False
        # self.has_validate_data = True if (validate_data or self.has_eval_data) else False

        if validate_data or (self.has_train_data and self.has_eval_data):
            self.has_validate_data = True

        if self.has_train_data and type(train_data) in ["DTable", "RDDTable"]:
            self.input_data_count = train_data.count()
        elif self.has_normal_input_data:
            for data_key, data_table in data.items():
                if type(data_table) in ["DTable", "RDDTable"]:
                    self.input_data_count = data_table.count()

        if self.has_validate_data and type(validate_data) in ["DTable", "RDDTable"]:
            self.input_eval_data_count = validate_data.count()

        self._abnormal_dsl_config_detect()
        LOGGER.debug(f"train_data: {train_data}, validate_data: {validate_data}, "
                     f"test_data: {test_data}, data: {data}")
        return train_data, validate_data, test_data, data

    def extract_running_rules(self, args, model):

        # train_data, eval_data, data = self.extract_input_data(args)
        train_data, validate_data, test_data, data = self.extract_input_data(args, model)

        running_funcs = RunningFuncs()
        schema = None
        for d in [train_data, validate_data, test_data]:
            if d is not None:
                schema = d.schema
                break

        if not self.need_run:
            running_funcs.add_func(model.pass_data, [data], save_result=True)
            return running_funcs

        if self.need_cv:
            running_funcs.add_func(model.cross_validation, [train_data])
            return running_funcs

        if self.need_stepwise:
            running_funcs.add_func(model.stepwise, [train_data], save_result=True)
            running_funcs.add_func(self.union_data, ["train"], use_previews=True, save_result=True)
            running_funcs.add_func(model.set_predict_data_schema, [schema],
                                   use_previews=True, save_result=True)
            return running_funcs

        if self.has_model or self.has_isometric_model:
            running_funcs.add_func(model.load_model, [args])

        if self.has_train_data and self.has_validate_data:
            # todo_func_list.extend([model.set_flowid, model.fit, model.set_flowid, model.predict])
            # todo_func_params.extend([['fit'], [train_data], ['validate'], [train_data, 'validate']])
            running_funcs.add_func(model.set_flowid, ['fit'])
            running_funcs.add_func(model.fit, [train_data, validate_data])
            running_funcs.add_func(model.set_flowid, ['validate'])
            running_funcs.add_func(model.predict, [train_data], save_result=True)
            running_funcs.add_func(model.set_flowid, ['predict'])
            running_funcs.add_func(model.predict, [validate_data], save_result=True)
            running_funcs.add_func(self.union_data, ["train", "validate"], use_previews=True, save_result=True)
            running_funcs.add_func(model.set_predict_data_schema, [schema],
                                   use_previews=True, save_result=True)

        elif self.has_train_data:
            running_funcs.add_func(model.set_flowid, ['fit'])
            running_funcs.add_func(model.fit, [train_data])
            running_funcs.add_func(model.set_flowid, ['validate'])
            running_funcs.add_func(model.predict, [train_data], save_result=True)
            running_funcs.add_func(self.union_data, ["train"], use_previews=True, save_result=True)
            running_funcs.add_func(model.set_predict_data_schema, [schema],
                                   use_previews=True, save_result=True)

        elif self.has_test_data:
            running_funcs.add_func(model.set_flowid, ['predict'])
            running_funcs.add_func(model.predict, [test_data], save_result=True)
            running_funcs.add_func(self.union_data, ["predict"], use_previews=True, save_result=True)
            running_funcs.add_func(model.set_predict_data_schema, [schema],
                                   use_previews=True, save_result=True)

        if self.has_normal_input_data and not self.has_model:
            running_funcs.add_func(model.extract_data, [data], save_result=True)
            running_funcs.add_func(model.set_flowid, ['fit'])
            running_funcs.add_func(model.fit, [], use_previews=True, save_result=True)

        if self.has_normal_input_data and self.has_model:
            running_funcs.add_func(model.extract_data, [data], save_result=True)
            running_funcs.add_func(model.set_flowid, ['transform'])
            running_funcs.add_func(model.transform, [], use_previews=True, save_result=True)

        return running_funcs

    @staticmethod
    def union_data(previews_data, name_list):
        if len(previews_data) == 0:
            return None

        if any([x is None for x in previews_data]):
            return None

        assert len(previews_data) == len(name_list)

        result_data = None
        for data, name in zip(previews_data, name_list):
            # LOGGER.debug("before mapValues, one data: {}".format(data.first()))
            data = data.mapValues(lambda value: value + [name])
            # LOGGER.debug("after mapValues, one data: {}".format(data.first()))

            if result_data is None:
                result_data = data
            else:
                LOGGER.debug(f"Before union, t1 count: {result_data.count()}, t2 count: {data.count()}")
                result_data = result_data.union(data)
                LOGGER.debug(f"After union, result count: {result_data.count()}")
            # LOGGER.debug("before out loop, one data: {}".format(result_data.first()))

        return result_data
