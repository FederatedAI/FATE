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


from .kfserving import get_kfserving_deployer


def model_deploy(party_model_id,
                 model_version,
                 model_object,
                 framework_name,
                 service_id,
                 deployment_type,
                 deployment_parameters):
    """
    Deploy a horizontally-trained model to a target serving system.

    Currently only KFServing is supported

    :param party_model_id: model id with party info to identify the model
    :param model_version: model version
    :param model_object: the converted model object
    :param framework_name: the ML framework of the converted model
    :param service_id: service name identifier
    :param deployment_type: currently only "kfserving" is supported
    :param deployment_parameters: parameters specific to the serving system
    :return: the deployed service representation, defined by the serving system.
    """
    if deployment_type == "kfserving":
        deployer = get_kfserving_deployer(party_model_id,
                                          model_version,
                                          model_object,
                                          framework_name,
                                          service_id,
                                          **deployment_parameters)
    else:
        raise ValueError("unknown deployment_type type: {}".format(deployment_type))
    return deployer.deploy()
