#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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

import importlib
import inspect
import os

from federatedml.util import LOGGER
from .component_converter import ComponentConverterBase

SKLEARN_FILENAME = "sklearn.joblib"
PYTORCH_FILENAME = "pytorch.pth"
TF_DIRNAME = "tensorflow_saved_model"
LGB_FILENAME = "lgb.txt"


def _get_component_converter(module_name: str,
                             framework_name: str):
    if framework_name in ["tensorflow", "tf", "tf_keras"]:
        framework_name = "tf_keras"
    elif framework_name in ["pytorch", "torch"]:
        framework_name = "pytorch"
    elif framework_name in ["sklearn", "scikit-learn"]:
        framework_name = "sklearn"
    elif framework_name in ['lightgbm']:
        framework_name = 'lightgbm'

    package_name = "." + framework_name
    parent_package = importlib.import_module(package_name, __package__)
    parent_package_path = os.path.dirname(os.path.realpath(parent_package.__file__))
    for f in os.listdir(parent_package_path):
        if f.startswith('.') or f.startswith('_'):
            continue
        if not f.endswith('.py'):
            continue
        proto_module = importlib.import_module("." + f.rstrip('.py'), parent_package.__name__)
        for name, obj in inspect.getmembers(proto_module):
            if inspect.isclass(obj) and issubclass(obj, ComponentConverterBase):
                for module in obj.get_target_modules():
                    if module.lower() == module_name.lower():
                        return framework_name, obj()
    return None, None


def get_default_target_framework(model_contents: dict,
                                 module_name: str):
    """
    Returns the name of a supported ML framework based on the
    original FATE model module name and model contents.

    :param model_contents: the model content of the FATE model
    :param module_name:  The module name, typically as HomoXXXX.
    :return: the corresponding framework name that this model can be converted to.
    """
    framework_name = None
    if module_name == "HomoLR":
        framework_name = "sklearn"
    elif module_name == 'HomoNN':
        # in FATE-1.10 currently support pytorch only
        framework_name = "pytorch"
        # if model_contents['HomoNNModelMeta'].params.config_type == "pytorch":
        #     framework_name = "pytorch"
        # else:
        #     framework_name = "tf_keras"
    elif module_name.lower() == 'homosecureboost':
        framework_name = 'lightgbm'
    else:
        LOGGER.debug(
            f"Module {module_name} is not a supported homogeneous model")
    return framework_name


def model_convert(model_contents: dict,
                  module_name: str,
                  framework_name=None):
    """Convert a Homo model component into format of a common ML framework

    :param model_contents: The model dict un-serialized from the model protobuf.
    :param module_name: The module name, typically as HomoXXXX.
    :param framework_name: The wanted framework, e.g. "sklearn", "pytorch", etc.
                           If not specified, the target framework will be chosen
                           automatically.
    :return: the converted framework name and a instance of the model object from
             the specified framework.
    """

    if not framework_name:
        framework_name = get_default_target_framework(
            model_contents, module_name)
        if not framework_name:
            return None, None
    target_framework, component_converter = _get_component_converter(
        module_name, framework_name)
    if not component_converter:
        LOGGER.warn(
            f"Module {module_name} cannot be converted to framework {framework_name}")
        return None, None
    LOGGER.info(
        f"Converting {module_name} module to a model of framework {target_framework}")

    return target_framework, component_converter.convert(model_contents)


def _get_model_saver_loader(framework_name: str):
    if framework_name in ["sklearn", "scikit-learn"]:
        import joblib
        return joblib.dump, joblib.load, SKLEARN_FILENAME
    elif framework_name in ["pytorch", "torch"]:
        import torch
        return torch.save, torch.load, PYTORCH_FILENAME
    elif framework_name in ["tensorflow", "tf", "tf_keras"]:
        import tensorflow
        return tensorflow.saved_model.save, tensorflow.saved_model.load, TF_DIRNAME
    elif framework_name in ['lightgbm']:
        from federatedml.protobuf.homo_model_convert.lightgbm.gbdt import save_lgb, load_lgb
        return save_lgb, load_lgb, LGB_FILENAME
    else:
        raise NotImplementedError("save method for framework: {} is not implemented"
                                  .format(framework_name))


def save_converted_model(model_object,
                         framework_name: str,
                         base_dir: str):
    """Save the model into target destination

    :param model_object: the model object
    :param framework_name: name of the framework of the model
    :param base_dir: the base directory to save the model file

    :return: local file/folder path
    """
    save, _, dest_filename = _get_model_saver_loader(framework_name)
    dest = os.path.join(base_dir, dest_filename)
    save(model_object, dest)
    LOGGER.info(f"Saved {framework_name} model to {dest}")
    return dest


def load_converted_model(framework_name: str,
                         base_dir: str):
    """Load a model from the specified directory previously used to save the converted model

    :param framework_name: name of the framework of the model
    :param base_dir: the base directory to save the model file

    :return: model object of the specified framework
    """
    _, load, src_filename = _get_model_saver_loader(framework_name)
    src = os.path.join(base_dir, src_filename)
    if not os.path.exists(src):
        raise FileNotFoundError(
            "expected file or folder {} doesn't exist".format(src))
    return load(src)
