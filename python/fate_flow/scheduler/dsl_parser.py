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
import importlib
import json

from fate_flow.settings import stat_logger
from fate_flow.utils.dsl_exception import DSLNotExistError, ComponentFieldNotExistError, \
    ModuleFieldNotExistError, ComponentInputTypeError, ComponentInputModelValueTypeError, \
    ModelInputComponentNotExistError, ModelInputNameNotExistError, ComponentInputDataTypeError, \
    ComponentInputDataValueTypeError, DataInputComponentNotExistError, DataInputNameNotExistError, \
    ComponentNotExistError, ModeError, DataNotExistInSubmitConfError, ComponentOutputTypeError, \
    ComponentOutputKeyTypeError, LoopError, ComponentMultiMappingError, NamingIndexError, \
    NamingError, NamingFormatError, DeployComponentNotExistError, ModuleNotExistError
from fate_flow.utils.runtime_conf_parse_util import RuntimeConfParserUtil


class Component(object):
    def __init__(self):
        self.module = None
        self.name = None
        self.upstream = []
        self.downstream = []
        self.role_parameters = {}
        self.input = {}
        self.output = {}
        self.component_provider = None

    def copy(self):
        copy_obj = Component()
        copy_obj.set_module(self.module)
        copy_obj.set_name(self.name)
        copy_obj.set_input(self.input)
        copy_obj.set_downstream(self.downstream)
        copy_obj.set_upstream(self.upstream)
        copy_obj.set_role_parameters(self.role_parameters)
        copy_obj.set_output(self.output)

        return copy_obj

    def set_input(self, inp):
        self.input = inp

    def get_input(self):
        return self.input

    def set_output(self, output):
        self.output = output

    def get_output(self):
        return self.output

    def get_module(self):
        return self.module

    def set_component_provider(self, interface):
        self.component_provider = interface

    def get_component_provider(self):
        return self.component_provider

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

    def set_role_parameters(self, role_parameters):
        self.role_parameters = role_parameters

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
        self.graph_dependency = None
        self.args_input = None
        self.args_data_key = None
        self.args_input_to_check = set()
        self.next_component_to_topo = set()
        self.job_parameters = {}
        self.provider_cache = {}
        self.job_providers = {}
        self.version = 2
        self.local_role = None
        self.local_party_id = None
        self.predict_runtime_conf = {}

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

                                idx_dependency = self.component_name_index.get(module_name)
                                self.component_downstream[idx_dependency].append(name)
                                self.component_upstream[idx].append(module_name)

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

                            idx_dependency = self.component_name_index.get(module_name)
                            self.component_downstream[idx_dependency].append(name)
                            self.component_upstream[idx].append(module_name)

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

    def _init_component_setting(self,
                                component,
                                provider_detail,
                                provider_name,
                                provider_version,
                                local_role,
                                local_party_id,
                                runtime_conf,
                                redundant_param_check=True):
        """
        init top input
        """
        pos = self.component_name_index[component]
        module = self.components[pos].get_module()

        parent_path = [component]
        cur_component = component
        while True:
            if self.train_input_model.get(cur_component, None) is None:
                break
            else:
                cur_component = self.train_input_model.get(cur_component)
                parent_path.append(cur_component)

        provider = RuntimeConfParserUtil.instantiate_component_provider(provider_detail,
                                                                        provider_name=provider_name,
                                                                        provider_version=provider_version)

        role_parameters = RuntimeConfParserUtil.get_component_parameters(provider,
                                                                         runtime_conf,
                                                                         module,
                                                                         cur_component,
                                                                         redundant_param_check=redundant_param_check,
                                                                         conf_version=self.version,
                                                                         local_role=local_role,
                                                                         local_party_id=local_party_id)

        for component in parent_path:
            idx = self.component_name_index.get(component)
            self.components[idx].set_component_provider(provider)
            self.components[idx].set_role_parameters(role_parameters)

        return role_parameters

    def parse_component_parameters(self, *args, **kwargs):
        raise NotImplementedError

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

        self.graph_dependency = base_dependency
        # self.graph_dependency = self.extract_need_run_status(base_dependency, component_parameters)

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

    def get_dependency(self):
        return self.graph_dependency

    def get_dependency_with_parameters(self, component_parameters):
        return self.extract_need_run_status(self.graph_dependency, component_parameters)

    def extract_need_run_status(self, graph_dependency, component_parameters):
        for rank in range(len(self.topo_rank)):
            idx = self.topo_rank[rank]
            name = self.components[idx].get_name()
            parameters = component_parameters.get(name)

            if not parameters:
                graph_dependency["component_need_run"][name] = False
            else:
                if self.train_input_model.get(name, None) is None:
                    param_name = "ComponentParam"
                    if parameters.get(param_name) is None \
                        or parameters[param_name].get("need_run") is False:
                        graph_dependency["component_need_run"][name] = False
                    else:
                        graph_dependency["component_need_run"][name] = True
                else:
                    input_model_name = self.train_input_model.get(name)
                    graph_dependency["component_need_run"][name] = graph_dependency["component_need_run"][
                            input_model_name]

        return graph_dependency

    @staticmethod
    def verify_dsl(dsl, mode="train"):
        raise NotImplementedError("verify dsl interface should be implemented")

    @staticmethod
    def deploy_component(*args, **kwargs):
        raise NotImplementedError

    def _auto_deduction(self, deploy_cpns=None, version=1, erase_top_data_input=False):
        self.predict_dsl = {"components": {}}
        self.predict_components = []
        mapping_list = {}
        for i in range(len(self.topo_rank)):
            self.predict_components.append(self.components[self.topo_rank[i]].copy())
            mapping_list[self.predict_components[-1].get_name()] = i

        output_data_maps = {}
        for i in range(len(self.predict_components)):
            name = self.predict_components[i].get_name()
            module = self.predict_components[i].get_module()

            if module == "Reader":
                if version != 2:
                    raise ValueError("Reader component can only be set in dsl_version 2")

            if self.get_need_deploy_parameter(name=name,
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
            if key not in self.args_data_key:
                raise DataNotExistInSubmitConfError(msg=key)

    def get_runtime_conf(self):
        return self.runtime_conf

    def get_dsl(self):
        return self.dsl

    def get_args_input(self):
        return self.args_input

    def get_need_deploy_parameter(self, name, deploy_cpns=None):
        if deploy_cpns is not None:
            return name in deploy_cpns

        return False

    def get_job_parameters(self):
        return self.job_parameters

    def get_job_providers(self, provider_detail=None, local_role=None, local_party_id=None):
        if self.job_providers:
            return self.job_providers
        else:
            self.job_providers = RuntimeConfParserUtil.get_job_providers(self.dsl, provider_detail, local_role,
                                                                         local_party_id, self.job_parameters)

            return self.job_providers

    @staticmethod
    def _gen_predict_data_mapping():
        return None, None

    @staticmethod
    def generate_predict_conf_template(train_dsl, train_conf, model_id, model_version):
        raise NotImplementedError


class DSLParser(BaseDSLParser):
    def __init__(self):
        super(DSLParser, self).__init__()
        self.version = 1

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
            provider_detail=None, mode="train", local_role=None,
            local_party_id=None, *args, **kwargs):

        if mode not in ["train", "predict"]:
            raise ModeError("")

        self.dsl = copy.deepcopy(dsl)
        self._init_components(mode)
        self._find_dependencies(mode)
        self.runtime_conf = runtime_conf
        self.pipeline_runtime_conf = pipeline_runtime_conf
        self.mode = mode
        self.local_role = local_role
        self.local_party_id = local_party_id

        if mode == "train":
            self.job_parameters = RuntimeConfParserUtil.get_job_parameters(self.runtime_conf,
                                                                           conf_version=1)

        elif mode == "predict":
            predict_runtime_conf = RuntimeConfParserUtil.merge_dict(pipeline_runtime_conf, runtime_conf)
            self.predict_runtime_conf = predict_runtime_conf
            self.job_parameters = RuntimeConfParserUtil.get_job_parameters(predict_runtime_conf,
                                                                           conf_version=1)

        self.args_input, self.args_data_key = RuntimeConfParserUtil.get_input_parameters(runtime_conf,
                                                                                         conf_version=1)
        self._check_args_input()

        self.prepare_graph_dependency_info()

        return self.components

    def generate_predict_dsl(self, deploy_detail):
        if deploy_detail:
            deploy_cpns = set()
            for component, need_deploy in deploy_detail.items():
                if need_deploy:
                    deploy_cpns.add(component)

            self._auto_deduction(deploy_cpns)

        return self.predict_dsl

    def _check_args_input(self):
        for key in self.args_input_to_check:
            if key not in self.args_data_key:
                raise DataNotExistInSubmitConfError(msg=key)

    def get_component_need_deploy_info(self, component, provider_detail, job_providers):
        if "need_deploy" in self.dsl["components"][component]:
            return self.dsl["components"][component].get("need_deploy")

        provider_info = job_providers[component]["provider"]
        name = provider_info["name"]
        version = provider_info["version"]
        provider = RuntimeConfParserUtil.instantiate_component_provider(provider_detail,
                                                                        provider_name=name,
                                                                        provider_version=version)

        module = self.dsl["components"][component].get("module")
        if hasattr(provider, "need_deploy"):
            return provider.need_deploy(module)

        return True

    def get_predict_dsl(self, predict_dsl=None, component_parameters=None):
        if predict_dsl is not None:
            return predict_dsl
        return self.add_module_info_to_predict_dsl(component_parameters)

    def add_module_info_to_predict_dsl(self, component_parameters):
        if not self.predict_dsl:
            return self.predict_dsl

        component_list = list(self.predict_dsl.get("components").keys())
        for component in component_list:
            parameters = component_parameters.get(component)
            if parameters:
                self.predict_dsl["components"][component]["CodePath"] = parameters.get("CodePath")

        return self.predict_dsl

    def parse_component_parameters(self, component_name, provider_detail, provider_name,
                                   provider_version, local_role, local_party_id):
        if self.mode == "predict":
            runtime_conf = self.predict_runtime_conf
            redundant_param_check = False
        else:
            runtime_conf = self.runtime_conf
            redundant_param_check = True

        parameters = self._init_component_setting(component_name,
                                                  provider_detail,
                                                  provider_name,
                                                  provider_version,
                                                  local_role,
                                                  local_party_id,
                                                  runtime_conf,
                                                  redundant_param_check)

        return parameters

    @staticmethod
    def _gen_predict_data_mapping():
        data_mapping = [("data", "data"), ("train_data", "test_data"),
                        ("validate_data", "test_data"), ("test_data", "test_data")]

        for data_key, data_value in data_mapping:
            yield data_key, data_value

    @staticmethod
    def generate_predict_conf_template(train_dsl, train_conf, model_id, model_version):
        return RuntimeConfParserUtil.generate_predict_conf_template(train_dsl,
                                                                    train_conf,
                                                                    model_id,
                                                                    model_version,
                                                                    conf_version=1)


class DSLParserV2(BaseDSLParser):
    def __init__(self):
        super(DSLParserV2, self).__init__()
        self.version = 2

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
            provider_detail=None, mode="train",
            local_role=None, local_party_id=None, *args, **kwargs):

        if mode not in ["train", "predict"]:
            raise ModeError("")

        self.dsl = copy.deepcopy(dsl)
        self._init_components(mode, version=2)
        self._find_dependencies(mode, version=2)
        self.runtime_conf = runtime_conf
        self.pipeline_runtime_conf = pipeline_runtime_conf
        self.mode = mode
        self.local_role = local_role
        self.local_party_id = local_party_id

        if mode == "train":
            self.job_parameters = RuntimeConfParserUtil.get_job_parameters(self.runtime_conf,
                                                                           conf_version=2)

        else:
            predict_runtime_conf = RuntimeConfParserUtil.merge_dict(pipeline_runtime_conf, runtime_conf)
            self.predict_runtime_conf = predict_runtime_conf
            self.job_parameters = RuntimeConfParserUtil.get_job_parameters(predict_runtime_conf,
                                                                           conf_version=2)

        self.args_input = RuntimeConfParserUtil.get_input_parameters(runtime_conf,
                                                                     components=self._get_reader_components(),
                                                                     conf_version=2)

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

    @staticmethod
    def get_job_providers_by_conf(dsl, runtime_conf, provider_detail,
                                  local_role, local_party_id, predict_conf=None):
        if not predict_conf:
            job_parameters = RuntimeConfParserUtil.get_job_parameters(runtime_conf,
                                                                      conf_version=2)
        else:
            train_job_parameters = RuntimeConfParserUtil.get_job_parameters(runtime_conf,
                                                                            conf_version=2)
            predict_job_parameters = RuntimeConfParserUtil.get_job_parameters(runtime_conf,
                                                                              conf_version=2)
            job_parameters = RuntimeConfParserUtil.merge_dict(predict_job_parameters, train_job_parameters)

        job_providers = RuntimeConfParserUtil.get_job_providers(dsl, provider_detail, local_role,
                                                                local_party_id, job_parameters)

        return job_providers

    @staticmethod
    def get_module_object_name(module, local_role, provider_detail,
                               provider_name, provider_version):
        if not provider_detail:
            raise ValueError("Component Providers should be provided")

        provider = RuntimeConfParserUtil.instantiate_component_provider(provider_detail,
                                                                        provider_name=provider_name,
                                                                        provider_version=provider_version)
        module_obj_name = RuntimeConfParserUtil.get_module_name(role=local_role,
                                                                module=module,
                                                                provider=provider)

        return module_obj_name

    @staticmethod
    def get_predict_dsl(predict_dsl=None, module_object_dict=None):
        if not predict_dsl:
            return {}

        role_predict_dsl = copy.deepcopy(predict_dsl)
        component_list = list(predict_dsl.get("components").keys())

        for component in component_list:
            module_object = module_object_dict.get(component)
            if module_object:
                role_predict_dsl["components"][component]["CodePath"] = module_object

            return role_predict_dsl

    def parse_component_parameters(self, component_name, provider_detail, provider_name, provider_version, local_role, local_party_id):
        if self.mode == "predict":
            runtime_conf = self.predict_runtime_conf
        else:
            runtime_conf = self.runtime_conf

        redundant_param_check = True
        parameters = self._init_component_setting(component_name,
                                                  provider_detail,
                                                  provider_name,
                                                  provider_version,
                                                  local_role,
                                                  local_party_id,
                                                  runtime_conf,
                                                  redundant_param_check)

        return parameters

    @staticmethod
    def _gen_predict_data_mapping():
        data_mapping = [("data", "data"), ("train_data", "test_data"),
                        ("validate_data", "test_data"), ("test_data", "test_data")]

        for data_key, data_value in data_mapping:
            yield data_key, data_value

    @staticmethod
    def generate_predict_conf_template(train_dsl, train_conf, model_id, model_version):
        return RuntimeConfParserUtil.generate_predict_conf_template(train_dsl,
                                                                    train_conf,
                                                                    model_id,
                                                                    model_version,
                                                                    conf_version=2)
