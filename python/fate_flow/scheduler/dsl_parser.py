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

from fate_flow.settings import stat_logger
from fate_flow.utils import parameter_util
from fate_flow.utils.dsl_exception import *


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


class BaseDSLParser(object):
    def __init__(self):
        self.dsl = None
        self.mode = "train"
        self.components = []
        self.component_name_index = {}
        self.component_upstream = []
        self.component_downstream = []
        self.train_input_model = {}
        self.in_degree = []
        self.topo_rank = []
        self.predict_dsl = {}
        self.runtime_conf = {}
        self.pipeline_runtime_conf = {}
        self.setting_conf_prefix = None
        self.graph_dependency = None
        self.args_input = None
        self.args_datakey = None
        self.args_input_to_check = set()
        self.next_component_to_topo = set()
        self.job_parameters = {}

    def _init_components(self, mode="train", version=1, **kwargs):
        if not self.dsl:
            raise DSLNotExistError("")

        components = self.dsl.get("components")

        if components is None:
            raise ComponentFieldNotExistError()

        for name in components:
            if "module" not in components[name]:
                raise ModuleFieldNotExistError(component=name)

            module = components[name]["module"]

            new_component = Component()
            new_component.set_name(name)
            new_component.set_module(module)
            self.component_name_index[name] = len(self.component_name_index)
            self.components.append(new_component)

        if version == 2 or mode == "train":
            self._check_component_valid_names()

    def _check_component_valid_names(self):
        raise NotImplementedError

    def _find_dependencies(self, mode="train", version=1):
        self.component_downstream = [[] for i in range(len(self.components))]
        self.component_upstream = [[] for i in range(len(self.components))]

        components_details = self.dsl.get("components")

        components_output = self._find_outputs()

        for name in self.component_name_index.keys():
            idx = self.component_name_index.get(name)
            upstream_input = components_details.get(name).get("input")
            downstream_output = components_details.get(name).get("output", {})

            self.components[idx].set_output(downstream_output)
            if upstream_input is None:
                continue
            elif not isinstance(upstream_input, dict):
                raise ComponentInputTypeError(component=name)
            else:
                self.components[idx].set_input(upstream_input)

            input_model_keyword = ["model", "isometric_model"]
            if mode == "train":
                for model_key in input_model_keyword:
                    if model_key in upstream_input:
                        model_list = upstream_input.get(model_key)

                        if not isinstance(model_list, list):
                            raise ComponentInputModelValueTypeError(component=name, other_info=model_list)

                        for model in model_list:
                            module_name = model.split(".", -1)[0]
                            input_model_name = module_name.split(".")[-1][0]
                            if module_name in ["args", "pipeline"]:
                                module_name = model.split(".", -1)[1]

                            if module_name not in self.component_name_index:
                                raise ModelInputComponentNotExistError(component=name, input_model=module_name)
                            else:
                                if module_name not in components_output or "model" not in components_output[
                                    module_name]:
                                    raise ModelInputNameNotExistError(component=name, input_model=module_name,
                                                                      other_info=input_model_name)

                                idx_dependendy = self.component_name_index.get(module_name)
                                self.component_downstream[idx_dependendy].append(name)
                                self.component_upstream[idx].append(module_name)
                                # self.component_upstream_model_relation_set.add((name, module_name))

                                if model_key == "model":
                                    self.train_input_model[name] = module_name

            if "data" in upstream_input:
                data_dict = upstream_input.get("data")
                if not isinstance(data_dict, dict):
                    raise ComponentInputDataTypeError(component=name)

                for data_set in data_dict:
                    if not isinstance(data_dict.get(data_set), list):
                        raise ComponentInputDataValueTypeError(component=name, other_info=data_dict.get(data_set))

                    if version == 2 and data_set not in ["data", "train_data", "validate_data", "test_data",
                                                         "eval_data"]:
                        stat_logger.warning(
                            "DSLParser Warning: make sure that input data's data key should be in {}, but {} found".format(
                                ["data", "train_data", "validate_data", "test_data", "eval_data"], data_set))
                    for data_key in data_dict.get(data_set):
                        module_name = data_key.split(".", -1)[0]
                        input_data_name = data_key.split(".", -1)[-1]
                        if module_name in ["args", "pipeline"]:
                            self.args_input_to_check.add(input_data_name)
                            continue

                        if module_name not in self.component_name_index:
                            raise DataInputComponentNotExistError(component=name, input_data=module_name)
                        else:
                            if module_name not in components_output \
                                    or "data" not in components_output[module_name] \
                                    or input_data_name not in components_output[module_name]["data"]:
                                raise DataInputNameNotExistError(component=name, input_data=module_name,
                                                                 other_info=input_data_name)

                            idx_dependendy = self.component_name_index.get(module_name)
                            self.component_downstream[idx_dependendy].append(name)
                            self.component_upstream[idx].append(module_name)
                            # self.component_upstream_data_relation_set.add((name, module_name))

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

    def _init_component_setting(self, setting_conf_prefix, runtime_conf, version=1, redundant_param_check=True):
        """
        init top input
        """
        for i in range(len(self.topo_rank)):
            idx = self.topo_rank[i]
            name = self.components[idx].get_name()
            if self.train_input_model.get(name, None) is None:
                module = self.components[idx].get_module()
                if version == 1:
                    role_parameters = parameter_util.ParameterUtil.override_parameter(setting_conf_prefix,
                                                                                      runtime_conf,
                                                                                      module,
                                                                                      name,
                                                                                      redundant_param_check=redundant_param_check)
                else:
                    role_parameters = parameter_util.ParameterUtilV2.override_parameter(setting_conf_prefix,
                                                                                        runtime_conf,
                                                                                        module,
                                                                                        name,
                                                                                        redundant_param_check=redundant_param_check)

                self.components[idx].set_role_parameters(role_parameters)
            else:
                up_component = self.train_input_model.get(name)
                up_idx = self.component_name_index.get(up_component)
                self.components[idx].set_role_parameters(self.components[up_idx].get_role_parameters())

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

            """add abnormal detection"""
            component_name = self.components[idx].get_name()
            if self.in_degree[idx] != 0:
                raise DegreeNotZeroError(component_name)

            if component_name in self.next_component_to_topo:
                raise ComponentDuplicateError(component=component_name)
            self.next_component_to_topo.add(component_name)

            for i in range(len(self.component_downstream[idx])):
                downstream_idx = self.component_name_index.get(self.component_downstream[idx][i])

                self.in_degree[downstream_idx] -= 1

                if self.in_degree[downstream_idx] == 0:
                    next_components.append(self.components[downstream_idx])

        return next_components

    def get_component_info(self, component_name):
        if component_name not in self.component_name_index:
            raise ComponentNotExistError(component=component_name)

        idx = self.component_name_index.get(component_name)
        return self.components[idx]

    def get_upstream_dependent_components(self, component_name):
        dependent_component_names = self.get_component_info(component_name).get_upstream()
        dependent_components = []
        for up_cpn in dependent_component_names:
            up_cpn_idx = self.component_name_index.get(up_cpn)
            dependent_components.append(self.components[up_cpn_idx])

        return dependent_components

    def get_topology_components(self):
        topo_components = []
        for i in range(len(self.topo_rank)):
            topo_components.append(self.components[self.topo_rank[i]])

        return topo_components

    def _find_outputs(self):
        outputs = {}

        components_details = self.dsl.get("components")

        for name in self.component_name_index.keys():
            if "output" not in components_details.get(name):
                continue

            component_output = components_details.get(name).get("output")
            output_keys = ["data", "model"]

            if not isinstance(component_output, dict):
                raise ComponentOutputTypeError(component=name, other_info=component_output)

            for key in output_keys:
                if key not in component_output:
                    continue

                out_v = component_output.get(key)
                if not isinstance(out_v, list):
                    raise ComponentOutputKeyTypeError(component=name, other_info=key)

                if name not in outputs:
                    outputs[name] = {}

                outputs[name][key] = out_v

        return outputs

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
            stack = []
            vis = [False for i in range(len(self.components))]
            for i in range(len(self.components)):
                if vis[i]:
                    continue
                loops = []
                self._find_loop(i, vis, stack, loops)
                raise LoopError(loops)

    def _find_loop(self, u, vis, stack, loops):
        vis[u] = True
        stack.append(u)
        for down_name in self.component_downstream[u]:
            if loops:
                return

            v = self.component_name_index.get(down_name)

            if v not in stack:
                if not vis[v]:
                    self._find_loop(v, vis, stack, loops)
            else:
                index = stack.index(v)
                for node in stack[index:]:
                    loops.append(self.components[node].get_name())

                return

        stack.pop(-1)

    def prepare_graph_dependency_info(self):
        dependence_dict = {}
        component_module = {}
        for component in self.components:
            name = component.get_name()
            module = component.get_module()
            component_module[name] = module
            if not component.get_input():
                continue
            dependence_dict[name] = []
            inputs = component.get_input()
            if "data" in inputs:
                data_input = inputs["data"]
                for data_key, data_list in data_input.items():
                    for dataset in data_list:
                        up_component_name = dataset.split(".", -1)[0]
                        if up_component_name == "args":
                            continue
                        up_pos = self.component_name_index.get(up_component_name)
                        up_component = self.components[up_pos]
                        data_name = dataset.split(".", -1)[1]
                        if up_component.get_output().get("data"):
                            data_pos = up_component.get_output().get("data").index(data_name)
                        else:
                            data_pos = 0

                        if up_component_name == "args":
                            continue

                        if data_key == "data" or data_key == "train_data":
                            data_type = data_key
                        else:
                            data_type = "validate_data"

                        dependence_dict[name].append({"component_name": up_component_name,
                                                      "type": data_type,
                                                      "up_output_info": ["data", data_pos]})

            model_keyword = ["model", "isometric_model"]
            for model_key in model_keyword:
                if model_key in inputs:
                    model_input = inputs[model_key]
                    for model_dep in model_input:
                        up_component_name = model_dep.split(".", -1)[0]
                        if up_component_name == "pipeline":
                            continue

                        model_name = model_dep.split(".", -1)[1]
                        up_pos = self.component_name_index.get(up_component_name)
                        up_component = self.components[up_pos]
                        if up_component.get_output().get("model"):
                            model_pos = up_component.get_output().get("model").index(model_name)
                        else:
                            model_pos = 0
                        dependence_dict[name].append({"component_name": up_component_name,
                                                      "type": "model",
                                                      "up_output_info": ["model", model_pos]})

            if not dependence_dict[name]:
                del dependence_dict[name]

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

    def get_dsl_hierarchical_structure(self):
        max_depth = [0] * len(self.components)
        for idx in range(len(self.topo_rank)):
            vertex = self.topo_rank[idx]
            for down_name in self.component_downstream[vertex]:
                down_vertex = self.component_name_index.get(down_name)
                max_depth[down_vertex] = max(max_depth[down_vertex], max_depth[vertex] + 1)

        max_dep = max(max_depth)
        hierarchical_structure = [[] for i in range(max_dep + 1)]
        name_component_maps = {}

        for component in self.components:
            name = component.get_name()
            vertex = self.component_name_index.get(name)
            hierarchical_structure[max_depth[vertex]].append(name)

            name_component_maps[name] = component

        return name_component_maps, hierarchical_structure

    def get_dependency(self, role, party_id):
        if role not in self.graph_dependency:
            raise ValueError("role {} is unknown, can not extract component dependency".format(role))

        if party_id not in self.graph_dependency[role]:
            raise ValueError("party id {} is unknown, can not extract component dependency".format(party_id))

        return self.graph_dependency[role][party_id]

    @staticmethod
    def verify_dsl(dsl, mode="train"):
        raise NotImplementedError("verify dsl interface should be implemented")

    @staticmethod
    def deploy_component(*args, **kwargs):
        raise NotImplementedError

    def _auto_deduction(self, setting_conf_prefix=None, deploy_cpns=None, version=1, erase_top_data_input=False):
        self.predict_dsl = {"components": {}}
        self.predict_components = []
        mapping_list = {}
        for i in range(len(self.topo_rank)):
            self.predict_components.append(copy.deepcopy(self.components[self.topo_rank[i]]))
            mapping_list[self.predict_components[-1].get_name()] = i

        output_data_maps = {}
        for i in range(len(self.predict_components)):
            name = self.predict_components[i].get_name()
            module = self.predict_components[i].get_module()

            if module == "Reader":
                if version != 2:
                    raise ValueError("Reader component can only be set in dsl_version 2")

            if self.get_need_deploy_parameter(name=name,
                                              setting_conf_prefix=setting_conf_prefix,
                                              deploy_cpns=deploy_cpns):

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
                        for data_key, data_value in self._gen_predict_data_mapping():
                            if data_key in self.dsl["components"][name]["input"]["data"]:
                                data_set = self.dsl["components"][name]["input"]["data"].get(data_key)
                                self.predict_dsl["components"][name]["input"]["data"][data_value] = []
                                for input_data in data_set:
                                    if version == 1 and input_data.split(".")[0] == "args":
                                        new_input_data = "args.eval_data"
                                        self.predict_dsl["components"][name]["input"]["data"][data_value].append(
                                            new_input_data)
                                    elif version == 2 and input_data.split(".")[0] == "args":
                                        self.predict_dsl["components"][name]["input"]["data"][data_value].append(
                                            input_data)
                                    elif version == 2 and self.dsl["components"][input_data.split(".")[0]].get(
                                            "module") == "Reader":
                                        self.predict_dsl["components"][name]["input"]["data"][data_value].append(
                                            input_data)
                                    else:
                                        pre_name = input_data.split(".")[0]
                                        data_suffix = input_data.split(".")[1]
                                        if self.get_need_deploy_parameter(name=pre_name,
                                                                          setting_conf_prefix=setting_conf_prefix,
                                                                          deploy_cpns=deploy_cpns):
                                            self.predict_dsl["components"][name]["input"]["data"][data_value].append(
                                                input_data)
                                        else:
                                            self.predict_dsl["components"][name]["input"]["data"][data_value] = \
                                                output_data_maps[
                                                    pre_name][data_suffix]

                                break

                        if version == 2 and erase_top_data_input:
                            is_top_component = True
                            for data_key, data_set in self.predict_dsl["components"][name]["input"]["data"].items():
                                for input_data in data_set:
                                    cpn_alias = input_data.split(".")[0]
                                    if cpn_alias == "args":
                                        is_top_component = False
                                        break

                                    if cpn_alias in self.predict_dsl["components"]:
                                        is_top_component = False

                            if is_top_component:
                                del self.predict_dsl["components"][name]["input"]["data"]

            else:
                name = self.predict_components[i].get_name()
                input_data = None
                output_data = None

                if "input" in self.dsl["components"][name] and "data" in self.dsl["components"][name]["input"]:
                    input_data = self.dsl["components"][name]["input"].get("data")

                if "output" in self.dsl["components"][name] and "data" in self.dsl["components"][name]["output"]:
                    output_data = self.dsl["components"][name]["output"].get("data")

                if output_data is None or input_data is None:
                    continue

                output_data_maps[name] = {}
                output_data_str = output_data[0]
                if "train_data" in input_data or "eval_data" in input_data or "test_data" in input_data:
                    if "train_data" in input_data:
                        up_input_data = input_data.get("train_data")[0]
                    elif "eval_data" in input_data:
                        up_input_data = input_data.get("eval_data")[0]
                    else:
                        up_input_data = input_data.get("test_data")[0]
                elif "data" in input_data:
                    up_input_data = input_data.get("data")[0]
                else:
                    raise ValueError("train data or eval data or validate data or data should be set")

                up_input_data_component_name = up_input_data.split(".", -1)[0]
                if up_input_data_component_name == "args" or self.get_need_deploy_parameter(
                        name=up_input_data_component_name,
                        setting_conf_prefix=setting_conf_prefix,
                        deploy_cpns=deploy_cpns):
                    output_data_maps[name][output_data_str] = [up_input_data]
                elif self.components[
                    self.component_name_index.get(up_input_data_component_name)].get_module() == "Reader":
                    output_data_maps[name][output_data_str] = [up_input_data]
                else:
                    up_input_data_suf = up_input_data.split(".", -1)[-1]
                    output_data_maps[name][output_data_str] = output_data_maps[up_input_data_component_name][
                        up_input_data_suf]

    def run(self, *args, **kwargs):
        pass

    def _check_args_input(self):
        for key in self.args_input_to_check:
            if key not in self.args_datakey:
                raise DataNotExistInSubmitConfError(msg=key)

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

    def get_need_deploy_parameter(self, name, setting_conf_prefix=None, deploy_cpns=None):
        raise NotImplementedError

    def get_job_parameters(self):
        return self.job_parameters

    @staticmethod
    def _gen_predict_data_mapping():
        return None, None

    @staticmethod
    def generate_predict_conf_template(train_dsl, train_conf, model_id, model_version):
        raise NotImplementedError

    @staticmethod
    def validate_component_param(setting_conf_prefix, runtime_conf, component_name, module, version=1):
        util = parameter_util.ParameterUtil if version == 1 else parameter_util.ParameterUtilV2
        try:
            util.override_parameter(setting_conf_prefix,
                                    runtime_conf,
                                    module,
                                    component_name,
                                    redundant_param_check=True)
            return 0
        except Exception as e:
            raise ValueError(f"{e}")


class DSLParser(BaseDSLParser):
    def _check_component_valid_names(self):
        occur_times = {}
        max_index = {}
        min_index = {}
        name_module_mapping = {}

        for component in self.components:
            name = component.get_name()
            module = component.get_module()

            if len(name.split("_", -1)) == 1:
                raise NamingError(name)

            index = name.split("_", -1)[-1]
            name_prefix = "_".join(name.split("_", -1)[:-1])

            try:
                index = int(index)
            except Exception as e:
                raise NamingIndexError(component=name)

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
                    raise ComponentMultiMappingError(component=name_prefix)

        for name in occur_times:
            if occur_times.get(name) != max_index.get(name) + 1:
                raise NamingIndexError(component=name)

    def _load_json(self, json_path):
        json_dict = None
        with open(json_path, "r") as fin:
            json_dict = json.loads(fin.read())

        if json_path is None:
            raise ValueError("can not load json file!!!")

        return json_dict

    @staticmethod
    def verify_dsl(dsl, mode="train"):
        dsl_parser = DSLParser()
        dsl_parser.dsl = dsl
        dsl_parser._init_components(mode=mode, version=1)
        dsl_parser._find_dependencies(mode=mode, version=1)

    def run(self, pipeline_dsl=None, pipeline_runtime_conf=None, dsl=None, runtime_conf=None,
            setting_conf_prefix=None, mode="train", *args, **kwargs):

        if mode not in ["train", "predict"]:
            raise ModeError("")

        self.dsl = copy.deepcopy(dsl)
        self._init_components(mode)
        self._find_dependencies(mode)
        self.runtime_conf = runtime_conf
        self.pipeline_runtime_conf = pipeline_runtime_conf
        self.mode = mode
        self.setting_conf_prefix = setting_conf_prefix

        if mode == "train":
            self._init_component_setting(setting_conf_prefix, self.runtime_conf)
            self.job_parameters = parameter_util.ParameterUtil.get_job_parameters(self.runtime_conf)
        else:
            predict_runtime_conf = parameter_util.ParameterUtil.merge_dict(pipeline_runtime_conf, runtime_conf)
            self._init_component_setting(setting_conf_prefix, predict_runtime_conf, redundant_param_check=False)
            self.job_parameters = parameter_util.ParameterUtil.get_job_parameters(predict_runtime_conf)

        self.args_input, self.args_datakey = parameter_util.ParameterUtil.get_args_input(runtime_conf, module="args")
        self._check_args_input()

        if mode == "train":
            self._auto_deduction(setting_conf_prefix)

        self.prepare_graph_dependency_info()

        return self.components

    def _check_args_input(self):
        for key in self.args_input_to_check:
            if key not in self.args_datakey:
                raise DataNotExistInSubmitConfError(msg=key)

    def get_need_deploy_parameter(self, name, setting_conf_prefix=None, deploy_cpns=None):
        if deploy_cpns is not None:
            return name in deploy_cpns

        if "need_deploy" in self.dsl["components"][name]:
            return self.dsl["components"][name].get("need_deploy")

        module = self.dsl["components"][name].get("module")

        setting_conf_path = os.path.join(setting_conf_prefix, module + ".json")
        if not os.path.isfile(setting_conf_path):
            raise ModuleNotExistError(component=name, module=module)

        need_deploy = True
        with open(os.path.join(setting_conf_prefix, module + ".json"), "r") as fin:
            setting_dict = json.loads(fin.read())
            need_deploy = setting_dict.get("need_deploy", True)

        return need_deploy

    def get_predict_dsl(self, role, predict_dsl=None, setting_conf_prefix=None):
        if predict_dsl is not None:
            return predict_dsl
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

    @staticmethod
    def _gen_predict_data_mapping():
        data_mapping = [("data", "data"), ("train_data", "test_data"),
                        ("validate_data", "test_data"), ("test_data", "test_data")]

        for data_key, data_value in data_mapping:
            yield data_key, data_value

    @staticmethod
    def generate_predict_conf_template(predict_dsl, train_conf, model_id, model_version):
        if not train_conf.get("role") or not train_conf.get("initiator"):
            raise ValueError("role and initiator should be contain in job's trainconf")

        predict_conf = dict()
        predict_conf["initiator"] = train_conf.get("initiator")
        predict_conf["role"] = train_conf.get("role")

        predict_conf["job_parameters"] = train_conf.get("job_parameters", {})
        predict_conf["job_parameters"]["job_type"] = "predict"
        predict_conf["job_parameters"]["model_id"] = model_id
        predict_conf["job_parameters"]["model_version"] = model_version

        predict_conf["role_parameters"] = {}

        for role in predict_conf["role"]:
            if role not in ["guest", "host"]:
                continue

            args_input = set()
            for _, module_info in predict_dsl.get("components", {}).items():
                data_set = module_info.get("input", {}).get("data", {})
                for data_key in data_set:
                    for data in data_set[data_key]:
                        if data.split(".", -1)[0] == "args":
                            args_input.add(data.split(".", -1)[1])

            predict_conf["role_parameters"][role] = {"args": {"data": {}}}
            fill_template = {}
            for data_key in args_input:
                fill_template[data_key] = [{"name": "name_to_be_filled_" + str(i),
                                            "namespace": "namespace_to_be_filled_" + str(i)}
                                           for i in range(len(predict_conf["role"].get(role)))]

            predict_conf["role_parameters"][role] = {"args": {"data": fill_template}}

        return predict_conf

    @staticmethod
    def validate_component_param(setting_conf_prefix, runtime_conf, component_name, module, *args):
        return BaseDSLParser.validate_component_param(setting_conf_prefix,
                                                      runtime_conf,
                                                      component_name,
                                                      module,
                                                      version=1)


class DSLParserV2(BaseDSLParser):
    def _check_component_valid_names(self):
        for component in self.components:
            name = component.get_name()
            for chk in name:
                if chk.isalpha() or chk in ["_", "-"] or chk.isdigit():
                    continue
                else:
                    raise NamingFormatError(component=name)

    @staticmethod
    def verify_dsl(dsl, mode="train"):
        dsl_parser = DSLParserV2()
        dsl_parser.dsl = dsl
        dsl_parser._init_components(mode=mode, version=2)
        dsl_parser._find_dependencies(mode=mode, version=2)

    @staticmethod
    def deploy_component(components, train_dsl):
        training_cpns = set(train_dsl.get("components").keys())
        deploy_cpns = set(components)
        if len(deploy_cpns & training_cpns) != len(deploy_cpns):
            raise DeployComponentNotExistError(msg=deploy_cpns - training_cpns)

        dsl_parser = DSLParserV2()
        dsl_parser.dsl = train_dsl
        dsl_parser._init_components()
        dsl_parser._find_dependencies(version=2)
        dsl_parser._auto_deduction(deploy_cpns=deploy_cpns, version=2, erase_top_data_input=True)

        return dsl_parser.predict_dsl

    def run(self, pipeline_runtime_conf=None, dsl=None, runtime_conf=None,
            setting_conf_prefix=None, mode="train", *args, **kwargs):

        if mode not in ["train", "predict"]:
            raise ModeError("")

        self.dsl = copy.deepcopy(dsl)
        self._init_components(mode, version=2)
        self._find_dependencies(mode, version=2)
        self.runtime_conf = runtime_conf
        self.pipeline_runtime_conf = pipeline_runtime_conf
        self.mode = mode
        self.setting_conf_prefix = setting_conf_prefix

        if mode == "train":
            self._init_component_setting(setting_conf_prefix, self.runtime_conf, version=2)
            self.job_parameters = parameter_util.ParameterUtilV2.get_job_parameters(self.runtime_conf)
        else:
            predict_runtime_conf = parameter_util.ParameterUtilV2.merge_dict(pipeline_runtime_conf, runtime_conf)
            self._init_component_setting(setting_conf_prefix, predict_runtime_conf, version=2)
            self.job_parameters = parameter_util.ParameterUtilV2.get_job_parameters(predict_runtime_conf)

        self.args_input = parameter_util.ParameterUtilV2.get_input_parameters(runtime_conf,
                                                                              components=self._get_reader_components())

        self.prepare_graph_dependency_info()

        return self.components

    def _get_reader_components(self):
        reader_components = []
        for cpn, conf in self.dsl.get("components").items():
            if conf.get("module") == "Reader":
                reader_components.append(cpn)

        return reader_components

    def get_need_deploy_parameter(self, name, deploy_cpns=None, **kwargs):
        if deploy_cpns is not None:
            return name in deploy_cpns

        return False

    def get_predict_dsl(self, role, predict_dsl=None, setting_conf_prefix=None):
        if not predict_dsl:
            return {}

        role_predict_dsl = copy.deepcopy(predict_dsl)
        component_list = list(predict_dsl.get("components").keys())
        for component in component_list:
            code_path = parameter_util.ParameterUtilV2.get_code_path(role=role,
                                                                     module=predict_dsl["components"][component][
                                                                         "module"],
                                                                     module_alias=component,
                                                                     setting_conf_prefix=setting_conf_prefix)
            if code_path:
                role_predict_dsl["components"][component]["CodePath"] = code_path

        return role_predict_dsl

    @staticmethod
    def _gen_predict_data_mapping():
        data_mapping = [("data", "data"), ("train_data", "test_data"),
                        ("validate_data", "test_data"), ("test_data", "test_data")]

        for data_key, data_value in data_mapping:
            yield data_key, data_value

    @staticmethod
    def generate_predict_conf_template(predict_dsl, train_conf, model_id, model_version):
        if not train_conf.get("role") or not train_conf.get("initiator"):
            raise ValueError("role and initiator should be contain in job's trainconf")

        predict_conf = dict()
        predict_conf["dsl_version"] = 2
        predict_conf["role"] = train_conf.get("role")
        predict_conf["initiator"] = train_conf.get("initiator")

        predict_conf["job_parameters"] = train_conf.get("job_parameters", {})
        predict_conf["job_parameters"]["common"].update({"model_id": model_id,
                                                         "model_version": model_version,
                                                         "job_type": "predict"})

        predict_conf["component_parameters"] = {"role": {}}

        for role in predict_conf["role"]:
            if role not in ["guest", "host"]:
                continue

            reader_components = []
            for module_alias, module_info in predict_dsl.get("components", {}).items():
                if module_info["module"] == "Reader":
                    reader_components.append(module_alias)

            predict_conf["component_parameters"]["role"][role] = dict()
            fill_template = {}
            for idx, reader_alias in enumerate(reader_components):
                fill_template[reader_alias] = {"table": {"name": "name_to_be_filled_" + str(idx),
                                                         "namespace": "namespace_to_be_filled_" + str(idx)}}

            for idx in range(len(predict_conf["role"][role])):
                predict_conf["component_parameters"]["role"][role][str(idx)] = fill_template

        return predict_conf

    @staticmethod
    def validate_component_param(setting_conf_prefix, runtime_conf, component_name, module, *args):
        return BaseDSLParser.validate_component_param(setting_conf_prefix,
                                                      runtime_conf,
                                                      component_name,
                                                      module,
                                                      version=2)

