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


from .pytorch import TorchServeKFDeployer
from .sklearn import SKLearnV1KFDeployer, SKLearnV2KFDeployer
from .tensorflow import TFServingKFDeployer


def get_kfserving_deployer(party_model_id,
                           model_version,
                           model_object,
                           framework_name,
                           service_id,
                           protocol_version="v1",
                           **kwargs):
    """
    Returns a deployer for KFServing InferenceService

    Refer to KFServingDeployer and its sub-classes
    for supported kwargs.
    :param party_model_id: the model id with party info used to identify the model
    :param model_version: the model version
    :param model_object: the converted model object
    :param framework_name: the framework of the model_object
    :param service_id: name of the serving service, will be used in KFServing as the service name
    :param protocol_version: the protocol version, currently only scikit-learn model supports v2
    :param kwargs: keyword argument to initialize the deployer object
    :return: an instance of the subclass of the base KFServingDeployer
    """
    if framework_name in ['sklearn', 'scikit-learn']:
        if protocol_version == "v2":
            cls = SKLearnV2KFDeployer
        else:
            cls = SKLearnV1KFDeployer
    elif framework_name in ['pytorch', 'torch']:
        cls = TorchServeKFDeployer
    elif framework_name in ['tf_keras', 'tensorflow', 'tf']:
        cls = TFServingKFDeployer
    else:
        raise ValueError("unknown converted model framework: {}".format(framework_name))
    return cls(party_model_id, model_version, model_object, service_id, **kwargs)
