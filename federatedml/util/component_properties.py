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


class ComponentProperties(object):
    def __init__(self):
        self.need_cv = False
        self.need_run = False
        self.need_stepwise = False
        self.has_model = False
        self.has_isometric_model = False
        self.has_train_data = False
        self.has_eval_data = False
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
            if "train_data" in data_keys:
                self.has_train_data = True
                data_keys.remove("train_data")

            if "eval_data" in data_keys:
                self.has_eval_data = True
                data_keys.remove("eval_data")

            if len(data_keys) > 0:
                self.has_normal_input_data = True

        LOGGER.debug("has_train_data: {}, has_eval_data: {}, has_normal_data: {}".format(
            self.has_train_data, self.has_eval_data, self.has_normal_input_data
        ))
        self._abnormal_dsl_config_detect()
        return self

    def _abnormal_dsl_config_detect(self):
        class DSLConfigError(ValueError):
            pass

        if self.has_model:
            if self.has_train_data:
                raise DSLConfigError("train_data input and model input should not be "
                                     "configured simultaneously")
            if self.has_isometric_model:
                raise DSLConfigError("model and isometric_model should not be "
                                     "configured simultaneously")
            if not self.has_eval_data and not self.has_normal_input_data:
                raise DSLConfigError("When model has been set, either eval_data or "
                                     "data should be provided")
        if self.has_normal_input_data:
            if self.has_train_data or self.has_eval_data:
                raise DSLConfigError("When data input has been configured, train_data "
                                     "and eval_data should not be configured.")

        if self.need_cv or self.need_stepwise:
            if not self.has_train_data:
                raise DSLConfigError("Train_data should be configured in cross-validate "
                                     "task or stepwise task")
            if self.has_eval_data or self.has_normal_input_data:
                raise DSLConfigError("In cross-validate task or stepwise task, eval_data "
                                     "or data should not be configured")

            if self.has_model or self.has_isometric_model:
                raise DSLConfigError("In cross-validate task or stepwise task, model "
                                     "or isometric_model should not be configured")

        if not self.need_run:
            if self.has_train_data or self.has_eval_data:
                raise DSLConfigError("Need run is false. This is component support "
                                     "data input only. Train_data and eval_data should not "
                                     "be configured")

    def extract_input_data(self, args):
        data_sets = args.get("data")
        train_data = None
        eval_data = None
        data = {}
        if data_sets is None:
            return train_data, eval_data, data

        LOGGER.debug(f"Input data_sets: {data_sets}")

        for data_key, data_dict in data_sets.items():

            for data_type, d_table in data_dict.items():
                if data_type == "train_data" and d_table is not None:
                    if isinstance(d_table, list):
                        train_data = d_table[0]
                    else:
                        train_data = d_table
                    if train_data is not None:
                        self.input_data_count = train_data.count()
                elif data_type == 'eval_data' and d_table is not None:
                    if isinstance(d_table, list):
                        eval_data = d_table[0]
                    else:
                        eval_data = d_table
                    # eval_data = d_table[0]
                    if eval_data is not None:
                        self.input_eval_data_count = eval_data.count()
                else:
                    if d_table is not None:
                        if isinstance(d_table, list):
                            data[".".join([data_key, data_type])] = d_table[0]
                        else:
                            data[".".join([data_key, data_type])] = d_table

            # if data_sets[data_key].get("data", None):
            #     # data = data_sets[data_key]["data"]
            #     data[data_key] = data_sets[data_key]["data"]

        for data_key, data_table in data.items():
            if data_table is not None:
                self.input_data_count += data_table.count()

        return train_data, eval_data, data

    def extract_running_rules(self, args, model):

        train_data, eval_data, data = self.extract_input_data(args)

        running_funcs = RunningFuncs()

        schema = None
        for d in [train_data, eval_data]:
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
            if eval_data:
                LOGGER.warn("Validate data provided for Stepwise Module. It will not be used in model training.")
            return running_funcs

        if self.has_model or self.has_isometric_model:
            running_funcs.add_func(model.load_model, [args])

        if self.has_train_data and self.has_eval_data:
            # todo_func_list.extend([model.set_flowid, model.fit, model.set_flowid, model.predict])
            # todo_func_params.extend([['fit'], [train_data], ['validate'], [train_data, 'validate']])
            running_funcs.add_func(model.set_flowid, ['fit'])
            running_funcs.add_func(model.fit, [train_data, eval_data])
            running_funcs.add_func(model.set_flowid, ['validate'])
            running_funcs.add_func(model.predict, [train_data], save_result=True)
            running_funcs.add_func(model.set_flowid, ['predict'])
            running_funcs.add_func(model.predict, [eval_data], save_result=True)
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

        elif self.has_eval_data:
            running_funcs.add_func(model.set_flowid, ['predict'])
            running_funcs.add_func(model.predict, [eval_data], save_result=True)
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

        # LOGGER.debug("func list: {}, param list: {}, save_results: {}, use_previews: {}".format(
        #     running_funcs.todo_func_list, running_funcs.todo_func_params,
        #     running_funcs.save_result, running_funcs.use_previews_result
        # ))
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
