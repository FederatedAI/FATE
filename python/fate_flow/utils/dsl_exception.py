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


class BaseDSLException(Exception):
    def __init__(self, msg):
        self.msg = msg


class DSLNotExistError(BaseDSLException):
    def __str__(self):
        return "There are no dsl, please check if the role and party id are correct"


class SubmitConfNotExistError(Exception):
    def __str__(self):
        return "SubMitConf is None, does not exist"


class ComponentFieldNotExistError(Exception):
    def __str__(self):
        return "No components filed in dsl, please have a check!"


class ModuleException(Exception):
    def __init__(self, component=None, module=None, input_model=None, input_data=None, output_model=None,
                 output_data=None, other_info=None):
        self.component = component
        self.module = module
        self.input_model = input_model
        self.input_data = input_data
        self.output_data = output_data
        self.output_model = output_model
        self.other_info = other_info


class ComponentNotExistError(ModuleException):
    def __str__(self):
        return "Component {} does not exist, please have a check".format(self.component)


class ModuleFieldNotExistError(ModuleException):
    def __str__(self):
        return "Component {}, module field does not exist in dsl, please have a check".format(self.component)


class ModuleNotExistError(ModuleException):
    def __str__(self):
        return "Component {}, module {} does not exist under the fold federatedml.setting_conf".format(self.component,
                                                                                                       self.module)


class ModuleConfigError(ModuleException):
    def __str__(self):
        return "Component {}, module {} config error, message is {}".format(self.component, self.module,
                                                                            self.other_info[0])


class DataNotExistInSubmitConfError(BaseDSLException):
    def __str__(self):
        return "{} does not exist in submit conf's data, please check!".format(self.msg)


class DefaultRuntimeConfNotExistError(ModuleException):
    def __str__(self):
        return "Default runtime conf of component {}, module {}, does not exist".format(self.component, self.module)


class DefaultRuntimeConfNotJsonError(ModuleException):
    def __str__(self):
        return "Default runtime conf of component {}, module {} should be json format file, but error occur: {}".format(self.component, self.module, self.other_info)


class ModelInputComponentNotExistError(ModuleException):
    def __str__(self):
        return "Component {}'s model input {} does not exist".format(self.component, self.input_model)


class ModelInputNameNotExistError(ModuleException):
    def __str__(self):
        return "Component {}' s model input {}'s output model {} does not exist".format(self.component,
                                                                                        self.input_model,
                                                                                        self.other_info)


class ComponentInputTypeError(ModuleException):
    def __str__(self):
        return "Input of component {} should be dict".format(self.component)


class ComponentOutputTypeError(ModuleException):
    def __str__(self):
        return "Output of component {} should be dict, but {} does not match".format(self.component, self.other_info)


class ComponentInputDataTypeError(ModuleException):
    def __str__(self):
        return "Component {}'s input data type should be dict".format(self.component)


class ComponentInputDataValueTypeError(ModuleException):
    def __str__(self):
        return "Component {}'s input data type should be list, but {} not match".format(self.component, self.other_info)


class ComponentInputModelValueTypeError(ModuleException):
    def __str__(self):
        return "Component {}'s input model value type should be list, but {} not match".format(self.component,
                                                                                               self.other_info)


class ComponentOutputKeyTypeError(ModuleException):
    def __str__(self):
        return "Component {}'s output key {} value type should be list".format(self.component,
                                                                               self.other_info)


class DataInputComponentNotExistError(ModuleException):
    def __str__(self):
        return "Component {}'s data input {} does not exist".format(self.component, self.input_data)


class DataInputNameNotExistError(ModuleException):
    def __str__(self):
        return "Component {}' data input {}'s output data {} does not exist".format(self.component,
                                                                                    self.input_data,
                                                                                    self.other_info)


class ParameterException(Exception):
    def __init__(self, parameter, role=None, msg=None):
        self.parameter = parameter
        self.role = role
        self.msg = msg


class ParamClassNotExistError(ModuleException):
    def __str__(self):
        return "Component {}, module {}'s param class {} does not exist".format(self.component, self.module,
                                                                                self.other_info)


class RoleParameterNotListError(ParameterException):
    def __str__(self):
        return "Role {} role parameter {} should be list".format(self.role, self.parameter)


class RoleParameterNotConsistencyError(ParameterException):
    def __str__(self):
        return "Role {} role parameter {} should be a list of same length with roles".format(self.role, self.parameter)


class ParameterCheckError(ModuleException):
    def __str__(self):
        return "Component {}, module {}, does not pass component check, error msg is {}".format(self.component,
                                                                                                self.module,
                                                                                                self.other_info)


class RedundantParameterError(ParameterCheckError):
    def __str__(self):
        return "Component {}, module {}, has redundant parameter {}".format(self.component,
                                                                            self.module,
                                                                            self.other_info)


class ComponentDuplicateError(ModuleException):
    def __str__(self):
        return "Component {} is duplicated, running before".format(self.component)


class DegreeNotZeroError(ModuleException):
    def __str__(self):
        return "Component {}' in degree should be zero for topological sort".format(self.component)


class ModeError(BaseDSLException):
    def __str__(self):
        return "dsl' s mode should be train or predict"


class LoopError(Exception):
    def __init__(self, components=None):
        self.components = components

    def __str__(self):
        if self.components is not None:
            return "{} form a loop".format("->".join(self.components))
        else:
            return "component relationship forms a dependency loop"


class NamingError(ModuleException):
    def __str__(self):
        return "Component's name should be format of name_index, index is start from 0 " + \
               "and be consecutive for same module, {} is error".format(self.component)


class NamingIndexError(ModuleException):
    def __str__(self):
        return "Component {}'s index should be an integer start from 0".format(self.component)


class NamingFormatError(ModuleException):
    def __str__(self):
        return "Component name {}'is not illegal, it should be consits of letter, digit, '-'  and '_'".format(self.component)


class ComponentMultiMappingError(ModuleException):
    def __str__(self):
        return "Component prefix {} should be used for only one module, but another".format(self.component)


class DeployComponentNotExistError(BaseDSLException):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "components {} not exist in training dsl, can not deploy!!!".format(self.msg)

