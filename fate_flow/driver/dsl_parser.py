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
# DSL PARSER
# =============================================================================

import copy
import json
import os

from fate_flow.utils import parameter_util


class Component(object):
    def __init__(self):
        self.module = None
        self.name = None
        self.upstream = []
        self.downstream = []
        self.role_parameters = {}
        self.input = {}
        self.output = {}

    def set_input(self, input):
        self.input = input

    def get_input(self):
        return self.input

    def set_output(self, output):
        self.output = output

    def get_output(self):
        return self.output

    def get_module(self):
        return self.module

    def get_name(self):
        return self.name

    def get_upstream(self):
        return self.upstream

    def get_downstream(self):
        return self.downstream

    def set_name(self, name):
        self.name = name

    def set_module(self, module):
        self.module = module

    def set_upstream(self, upstream):
        self.upstream = upstream

    def set_downstream(self, downstream):
        self.downstream = downstream

    def set_role_parameters(self, role_parametes):
        self.role_parameters = role_parametes

    def get_role_parameters(self):
        return self.role_parameters


class DSLParser(object):
    def __init__(self):
        self.dsl = None
        self.mode = "train"
        self.components = []
        self.component_name_index = {}
        self.component_upstream = []
        self.component_downstream = []
        self.component_upstream_data_relation_set = set()
        self.component_upstream_model_relation_set = set()
        self.train_input_model = {}
        self.in_degree = []
        self.topo_rank = []
        self.predict_dsl = {}
        self.runtime_conf = {}
        self.pipeline_runtime_conf = {}
        self.pipeline_modules = {}
        self.pipeline_module_alias = None
        self.setting_conf_prefix = None
        self.graph_dependency = None
        self.args_input = None

    def _init_components(self, pipeline_dsl=None, mode="train"):
        if not self.dsl:
            raise Exception("there are no dsl, please check if the role and party id are correct")
        components = self.dsl.get("components")
        if components is None:
            raise ValueError("there are no components in dsl, please have a check!")

        pipeline_cnt = 0
        for name in components:
            module = components[name]["module"]
            if module == "Pipeline":
                if pipeline_dsl is None:
                    raise ValueError("find module Pipeline")
                pipeline_cnt += 1
                if pipeline_cnt > 1:
                    raise ValueError("pipeline module only support once")

                self.pipeline_module_alias = name

                pipeline_components = pipeline_dsl.get("components")
                for pipeline_component in pipeline_components:
                    pipeline_module = pipeline_components[pipeline_component]

                    new_component = Component()
                    new_component.set_name(pipeline_component)
                    new_component.set_module(pipeline_module)
                    self.component_name_index[pipeline_component] = len(self.component_name_index)
                    self.components.append(new_component)

                    self.pipeline_modules[pipeline_component] = True

                continue

            new_component = Component()
            new_component.set_name(name)
            new_component.set_module(module)
            self.component_name_index[name] = len(self.component_name_index)
            self.components.append(new_component)

        if mode == "train":
            self._check_component_valid_names()

    def _init_component_setting(self, setting_conf_prefix, runtime_conf, default_runtime_conf_prefix):
        """
        init top input
        """
        for i in range(len(self.topo_rank)):
            idx = self.topo_rank[i]
            name = self.components[idx].get_name()
            if self.train_input_model.get(name, None) is None:
                module = self.components[idx].get_module()
                role_parameters = parameter_util.ParameterUtil.override_parameter(default_runtime_conf_prefix,
                                                                                  setting_conf_prefix,
                                                                                  runtime_conf,
                                                                                  module,
                                                                                  name)

                self.components[idx].set_role_parameters(role_parameters)
            else:
                up_component = self.train_input_model.get(name)
                up_idx = self.component_name_index.get(up_component)
                self.components[idx].set_role_parameters(self.components[up_idx].get_role_parameters())

    def _check_component_valid_names(self):
        occur_times = {}
        max_index = {}
        min_index = {}
        name_module_mapping = {}

        for component in self.components:
            name = component.get_name()
            module = component.get_module()

            if len(name.split("_", -1)) == 1:
                raise ValueError("component's name should be format of name_index, index is start from 0 "
                                 "and be consecutive for same module")

            index = name.split("_", -1)[-1]
            name_prefix = "_".join(name.split("_", -1)[:-1])

            try:
                index = int(index)
            except:
                raise ValueError("index of component's name should be integer start from 0")

            if name_prefix not in occur_times:
                occur_times[name_prefix] = 1
                max_index[name_prefix] = index
                min_index[name_prefix] = index
                name_module_mapping[name_prefix] = module
            else:
                occur_times[name_prefix] += 1
                max_index[name_prefix] = max(max_index[name_prefix], index)
                min_index[name_prefix] = min(min_index[name_prefix], index)
                if name_module_mapping.get(name_prefix) != module:
                    raise ValueError("component's name should be format of name_index, index is start from 0 "
                                     "and same modules should have the same name")

        for name in occur_times:
            if occur_times.get(name) != max_index.get(name) + 1:
                raise ValueError("component's name should be format of name_index, index is start from 0 "
                                 "and be consecutive for same module")

    def _find_dependencies(self, pipeline_dsl=None, mode="train"):
        self.component_downstream = [[] for i in range(len(self.components))]
        self.component_upstream = [[] for i in range(len(self.components))]

        components_details = self.dsl.get("components")
        if pipeline_dsl is not None:
            pipeline_components_details = pipeline_dsl.get("components")
        else:
            pipeline_components_details = None

        for name in self.component_name_index.keys():
            if name in self.pipeline_modules:
                pipeline_module = True
            else:
                pipeline_module = False

            if pipeline_module:
                if pipeline_components_details.get("input") is None:
                    continue
            else:
                if components_details.get(name).get("input") is None:
                    continue

            idx = self.component_name_index.get(name)
            if not pipeline_module:
                upstream_input = components_details.get(name).get("input")
                downstream_output = components_details.get(name).get("output", {})
            else:
                upstream_input = pipeline_components_details.get(name).get("input")
                downstream_output = components_details.get(name).get("output", {})

            self.components[idx].set_input(upstream_input)
            self.components[idx].set_output(downstream_output)

            input_model_keyword = ["model", "isometric_model"]
            if mode == "train":
                for model_key in input_model_keyword:
                    if model_key in upstream_input:
                        model_list = upstream_input.get(model_key)
                        for model in model_list:
                            module_name = model.split(".", -1)[0]
                            if module_name in ["args", "pipeline"]:
                                module_name = model.split(".", -1)[1]

                            if module_name not in self.component_name_index:
                                raise ValueError("unknown module input {}".format(model))
                            else:
                                if module_name == self.pipeline_module_alias:
                                    raise ValueError("Pipeline Model can not be used")

                                if name not in self.pipeline_modules and module_name in self.pipeline_modules:
                                    raise ValueError("Pipeline Model can not be used")

                                idx_dependendy = self.component_name_index.get(module_name)
                                self.component_downstream[idx_dependendy].append(name)
                                self.component_upstream[idx].append(module_name)
                                self.component_upstream_model_relation_set.add((name, module_name))

                                if model_key == "model":
                                    self.train_input_model[name] = module_name

            if "data" in upstream_input:
                data_dict = upstream_input.get("data")
                for data_set in data_dict:
                    for data_key in data_dict.get(data_set):
                        module_name = data_key.split(".", -1)[0]
                        if module_name in ["args", "pipeline"]:
                            continue

                        if module_name == self.pipeline_module_alias:
                            if pipeline_dsl is None:
                                raise ValueError("If use pipeline, pipiline dsl should not be None")
                            end_data_module = self._get_end_of_pipeline_dsl(pipeline_dsl)
                            idx_dependendy = self.component_name_index.get(end_data_module)
                            self.component_downstream[idx_dependendy].append(name)
                            self.component_upstream[idx].append(end_data_module)
                            self.component_upstream_data_relation_set.add((name, end_data_module))

                        elif module_name not in self.component_name_index:
                            raise ValueError("unknown module input {}".format(module_name))
                        else:
                            if name not in self.pipeline_modules and module_name in self.pipeline_modules:
                                raise ValueError("Only can use pipeline's data in the end")

                            idx_dependendy = self.component_name_index.get(module_name)
                            self.component_downstream[idx_dependendy].append(name)
                            self.component_upstream[idx].append(module_name)
                            self.component_upstream_data_relation_set.add((name, module_name))

        self.in_degree = [0 for i in range(len(self.components))]
        for i in range(len(self.components)):
            if self.component_downstream[i]:
                self.component_downstream[i] = list(set(self.component_downstream[i]))

            if self.component_upstream[i]:
                self.component_upstream[i] = list(set(self.component_upstream[i]))
                self.in_degree[self.component_name_index.get(self.components[i].get_name())] = len(
                    self.component_upstream[i])

        self._check_dag_dependencies()

        for i in range(len(self.components)):
            self.components[i].set_upstream(self.component_upstream[i])
            self.components[i].set_downstream(self.component_downstream[i])

    def _check_dag_dependencies(self):
        in_degree = copy.deepcopy(self.in_degree)
        stack = []
        for i in range(len(self.components)):
            if in_degree[i] == 0:
                stack.append(i)

        tot_nodes = 0

        while len(stack) > 0:
            idx = stack.pop()
            tot_nodes += 1
            self.topo_rank.append(idx)

            for down_name in self.component_downstream[idx]:
                down_idx = self.component_name_index.get(down_name)
                in_degree[down_idx] -= 1

                if in_degree[down_idx] == 0:
                    stack.append(down_idx)

        if tot_nodes != len(self.components):
            raise ValueError("component dependencies form loop, please check")

    def _load_json(self, json_path):
        json_dict = None
        with open(json_path, "r") as fin:
            json_dict = json.loads(fin.read())

        if json_path is None:
            raise ValueError("can not load json file!!!")

        return json_dict

    def get_next_components(self, module_name=None):
        next_components = []
        if module_name is None:
            for i in range(len(self.components)):
                if self.in_degree[i] == 0:
                    next_components.append(self.components[i])
        else:
            if module_name not in self.component_name_index:
                raise ValueError("{} module_name not find!".format(module_name))

            idx = self.component_name_index.get(module_name)

            for i in range(len(self.component_downstream[idx])):
                downstream_idx = self.component_name_index.get(self.component_downstream[idx][i])

                self.in_degree[downstream_idx] -= 1

                if self.in_degree[downstream_idx] == 0:
                    next_components.append(self.components[downstream_idx])

        return next_components

    def get_component_info(self, component_name):
        idx = self.component_name_index.get(component_name)
        return self.components[idx]

    def _get_end_of_pipeline_dsl(self, pipeline_dsl):
        has_up = {}
        has_down = {}
        component_details = pipeline_dsl.get("components")
        for component in component_details:
            if component_details[component].get("input", None):
                model_input_keyword = ["model", "isometric_model"]
                for model_key in model_input_keyword:
                    if component_details[component]["input"].get(model_key, None):
                        model_list = component_details[component]["input"][model_key]
                        for model in model_list:
                            model_name = model.split(".", -1)
                            if model_name[0] == "args":
                                continue
                            has_down[model_name[0]] = True

                if component_details[component]["input"].get("data", None):
                    for data_set in component_details[component]["input"]["data"]:
                        for data_key in data_set:
                            model_name = data_key.split(".", -1)
                            if model_name[0] == "args":
                                continue
                            has_down[model_name[0]] = True

        for component in component_details:
            if not has_down[component]:
                if component_details[component].get("output", None):
                    if "data" in component_details[component]["output"]:
                        return component

        raise ValueError("No end data of pipeline dsl, check it plz")

    def prepare_graph_dependency_info(self):
        dependence_dict = {}
        component_module = {}
        for component in self.components:
            name = component.get_name()
            module = component.get_module()
            component_module[name] = module
            upstream = self.component_upstream[self.component_name_index.get(name)]
            if upstream:
                dependence_dict[name] = []
                for up_component in upstream:
                    if (name, up_component) in self.component_upstream_data_relation_set:
                        dependence_dict[name].append({"component_name": up_component,
                                                      "type": "data"})

                    if (name, up_component) in self.component_upstream_model_relation_set:
                        dependence_dict[name].append({"component_name": up_component,
                                                      "type": "model"})

        component_list = [None for i in range(len(self.components))]
        topo_rank_reverse_mapping = {}
        for i in range(len(self.topo_rank)):
            topo_rank_reverse_mapping[self.topo_rank[i]] = i

        for key, value in self.component_name_index.items():
            topo_rank_idx = topo_rank_reverse_mapping[value]
            component_list[topo_rank_idx] = key

        base_dependency = {"component_list": component_list,
                           "dependencies": dependence_dict,
                           "component_module": component_module,
                           "component_need_run": {}}

        if self.mode == "train":
            runtime_conf = self.runtime_conf
        else:
            runtime_conf = self.pipeline_runtime_conf

        self.graph_dependency = {}
        for role in runtime_conf["role"]:
            self.graph_dependency[role] = {}
            dependency_list = [copy.deepcopy(base_dependency) for i in range(len(runtime_conf["role"].get(role)))]

            for rank in range(len(self.topo_rank)):
                idx = self.topo_rank[rank]
                name = self.components[idx].get_name()
                module = self.components[idx].get_module()
                parameters = self.components[idx].get_role_parameters()

                if role not in parameters:
                    for i in range(len(dependency_list)):
                        dependency_list[i]["component_need_run"][name] = False
                else:
                    if self.train_input_model.get(name, None) is None:
                        param_class = parameter_util.ParameterUtil.get_param_class_name(self.setting_conf_prefix,
                                                                                        module)
                        for i in range(len(dependency_list)):
                            if parameters[role][i].get(param_class) is None \
                                    or parameters[role][i][param_class].get("need_run") is False:
                                dependency_list[i]["component_need_run"][name] = False
                            else:
                                dependency_list[i]["component_need_run"][name] = True
                    else:
                        input_model_name = self.train_input_model.get(name)
                        for i in range(len(dependency_list)):
                            dependency_list[i]["component_need_run"][name] = dependency_list[i]["component_need_run"][
                                input_model_name]

            for i in range(len(runtime_conf["role"].get(role))):
                party_id = runtime_conf["role"].get(role)[i]
                self.graph_dependency[role][party_id] = dependency_list[i]

    def get_dependency(self, role, party_id):
        if role not in self.graph_dependency:
            raise ValueError("role {} is unknown, can not extract component dependency".format(role))

        if party_id not in self.graph_dependency[role]:
            raise ValueError("party id {} is unknown, can not extract component dependency".format(party_id))

        return self.graph_dependency[role][party_id]

    def _auto_deduction(self, setting_conf_prefix):
        self.predict_dsl = {"components": {}}
        self.predict_components = []
        mapping_list = {}
        for i in range(len(self.topo_rank)):
            self.predict_components.append(copy.deepcopy(self.components[self.topo_rank[i]]))
            mapping_list[self.predict_components[-1].get_name()] = i

        need_predict = False
        output_data_maps = {}
        for i in range(len(self.predict_components)):
            name = self.predict_components[i].get_name()
            if self.get_need_deploy_parameter(name=name,
                                              setting_conf_prefix=setting_conf_prefix):
                need_predict = True
                self.predict_dsl["components"][name] = {"module": self.predict_components[i].get_module()}

                """replace output model to pippline"""
                if "output" in self.dsl["components"][name]:
                    model_list = self.dsl["components"][name]["output"].get("model", None)
                    if model_list is not None:
                        if "input" not in self.predict_dsl["components"][name]:
                            self.predict_dsl["components"][name]["input"] = {}

                        replace_model = []
                        for model in model_list:
                            replace_str = ".".join(["pipeline", name, model])
                            replace_model.append(replace_str)

                        self.predict_dsl["components"][name]["input"]["model"] = replace_model

                    output_data = copy.deepcopy(self.dsl["components"][name]["output"].get("data", None))
                    if output_data is not None:
                        if "output" not in self.predict_dsl["components"][name]:
                            self.predict_dsl["components"][name]["output"] = {}

                        self.predict_dsl["components"][name]["output"]["data"] = output_data

                if "input" in self.dsl["components"][name]:
                    if "input" not in self.predict_dsl["components"][name]:
                        self.predict_dsl["components"][name]["input"] = {}
                    if "data" in self.dsl["components"][name]["input"]:
                        self.predict_dsl["components"][name]["input"]["data"] = {}
                        if "data" in self.dsl["components"][name]["input"]["data"]:
                            data_set = self.dsl["components"][name]["input"]["data"].get("data")
                            self.predict_dsl["components"][name]["input"]["data"]["data"] = []
                            for input_data in data_set:
                                if input_data.split(".")[0] == "args":
                                    new_input_data = "args.eval_data"
                                    self.predict_dsl["components"][name]["input"]["data"]["data"].append(new_input_data)
                                else:
                                    pre_name = input_data.split(".")[0]
                                    data_suffix = input_data.split(".")[1]
                                    pre_idx = mapping_list.get(pre_name)
                                    if self.get_need_deploy_parameter(name=pre_name,
                                                                      setting_conf_prefix=setting_conf_prefix):
                                        self.predict_dsl["components"][name]["input"]["data"]["data"].append(input_data)
                                    else:
                                        self.predict_dsl["components"][name]["input"]["data"]["data"] = \
                                            output_data_maps[
                                                pre_name][data_suffix]

                        elif "train_data" in self.dsl["components"][name]["input"]["data"]:
                            input_data = self.dsl["components"][name]["input"]["data"].get("train_data")[0]
                            if input_data.split(".")[0] == "args":
                                new_input_data = "args.eval_data"
                                self.predict_dsl["components"][name]["input"]["data"]["eval_data"] = [new_input_data]
                            else:
                                pre_name = input_data.split(".")[0]
                                data_suffix = input_data.split(".")[1]
                                pre_idx = mapping_list.get(pre_name)
                                if self.get_need_deploy_parameter(name=pre_name,
                                                                  setting_conf_prefix=setting_conf_prefix):

                                    # if self.dsl["components"][pre_name].get("need_deploy", None):
                                    self.predict_dsl["components"][name]["input"]["data"]["eval_data"] = [input_data]
                                else:
                                    self.predict_dsl["components"][name]["input"]["data"]["eval_data"] = \
                                        output_data_maps[
                                            pre_name][data_suffix]

                        elif "eval_data" in self.dsl["components"][name]["input"]["data"]:
                            input_data = self.dsl["components"][name]["input"]["data"].get("eval_data")[0]
                            if input_data.split(".")[0] == "args":
                                new_input_data = "args.eval_data"
                                self.predict_dsl["components"][name]["input"]["data"]["eval_data"] = new_input_data
                            else:
                                pre_name = input_data.split(".")[0]
                                data_suffix = input_data.split(".")[1]
                                pre_idx = mapping_list.get(pre_name)
                                if self.get_need_deploy_parameter(name=pre_name,
                                                                  setting_conf_prefix=setting_conf_prefix):
                                    self.predict_dsl["components"][name]["input"]["data"]["eval_data"] = [input_data]
                                else:
                                    self.predict_dsl["components"][name]["input"]["data"]["eval_data"] = \
                                        output_data_maps[
                                            pre_name].get(data_suffix)

            else:
                module = self.predict_components[i].get_module()
                name = self.predict_components[i].get_name()
                input_data = None
                output_data = None

                if "input" in self.dsl["components"][name] and "data" in self.dsl["components"][name]["input"]:
                    input_data = self.dsl["components"][name]["input"].get("data")

                if "output" in self.dsl["components"][name] and "data" in self.dsl["components"][name]["output"]:
                    output_data = self.dsl["components"][name]["output"].get("data")

                if output_data is None:
                    continue

                output_data_maps[name] = {}
                output_data_str = output_data[0]
                if "train_data" in input_data or "eval_data" in input_data:
                    if "train_data" in input_data:
                        up_input_data = input_data.get("train_data")[0]
                    else:
                        up_input_data = input_data.get("eval_data")[0]
                elif "data" in input_data:
                    up_input_data = input_data.get("data")[0]
                else:
                    raise ValueError("Illegal input data")

                up_input_data_component_name = up_input_data.split(".", -1)[0]
                if up_input_data_component_name == "args" or self.get_need_deploy_parameter(
                        name=up_input_data_component_name,
                        setting_conf_prefix=setting_conf_prefix):
                    output_data_maps[name][output_data_str] = [up_input_data]
                else:
                    up_input_data_suf = up_input_data.split(".", -1)[-1]
                    output_data_maps[name][output_data_str] = output_data_maps[up_input_data_component_name][
                        up_input_data_suf]

        if not need_predict:
            return

    def run(self, pipeline_dsl=None, pipeline_runtime_conf=None, dsl=None, runtime_conf=None,
            default_runtime_conf_prefix=None,
            setting_conf_prefix=None, mode="train"):

        self.dsl = copy.deepcopy(dsl)
        self._init_components(pipeline_dsl, mode)
        self._find_dependencies(pipeline_dsl, mode)
        self.runtime_conf = runtime_conf
        self.pipeline_runtime_conf = pipeline_runtime_conf
        self.mode = mode
        self.setting_conf_prefix = setting_conf_prefix

        if mode == "train":
            self._init_component_setting(setting_conf_prefix, self.runtime_conf, default_runtime_conf_prefix)
        else:
            self._init_component_setting(setting_conf_prefix, pipeline_runtime_conf, default_runtime_conf_prefix)

        self.args_input = parameter_util.ParameterUtil.get_args_input(runtime_conf, module="args")

        if mode == "train":
            self._auto_deduction(setting_conf_prefix)

        self.prepare_graph_dependency_info()

        return self.components

    def get_need_deploy_parameter(self, name, setting_conf_prefix):
        if "need_deploy" in self.dsl["components"][name]:
            return self.dsl["components"][name].get("need_deploy")

        module = self.dsl["components"][name].get("module")
        need_deploy = True
        with open(os.path.join(setting_conf_prefix, module + ".json"), "r") as fin:
            setting_dict = json.loads(fin.read())
            need_deploy = setting_dict.get("need_deploy", True)

        return need_deploy

    def get_predict_dsl(self, role):
        return self.gen_predict_dsl_by_role(role)

    def gen_predict_dsl_by_role(self, role):
        if not self.predict_dsl:
            return self.predict_dsl

        role_predict_dsl = copy.deepcopy(self.predict_dsl)
        component_list = list(self.predict_dsl.get("components").keys())
        for component in component_list:
            idx = self.component_name_index.get(component)
            role_parameters = self.components[idx].get_role_parameters()
            if role in role_parameters:
                role_predict_dsl["components"][component]["CodePath"] = role_parameters[role][0].get("CodePath")

        return role_predict_dsl

    def get_runtime_conf(self):
        return self.runtime_conf

    def get_dsl(self):
        return self.dsl

    def get_args_input(self):
        return self.args_input
