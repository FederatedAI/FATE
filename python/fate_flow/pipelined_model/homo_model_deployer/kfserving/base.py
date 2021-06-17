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

import io
import uuid
import kfserving
from kfserving.api import creds_utils
from kubernetes import client

from fate_flow.settings import stat_logger
from .model_storage import get_model_storage, ModelStorageType
from .model_storage.minio import MinIOModelStorage

MINIO_K8S_SECRET_NAME = "fate-homo-serving-minio-secret"

STORAGE_URI_KEY = "storage_uri"

ANNOTATION_PREFIX = "fate.fedai.org/"
ANNOTATION_SERVICE_UUID = ANNOTATION_PREFIX + "uuid"
ANNOTATION_FATE_MODEL_ID = ANNOTATION_PREFIX + "model_id"
ANNOTATION_FATE_MODEL_VERSION = ANNOTATION_PREFIX + "model_version"


class KFServingDeployer(object):
    """Class representing a KFServing service deployer
    """

    def __init__(self,
                 party_model_id,
                 model_version,
                 model_object,
                 service_id,
                 namespace=None,
                 config_file_content=None,
                 replace=False,
                 skip_create_storage_secret=False,
                 model_storage_type=ModelStorageType.MINIO,
                 model_storage_parameters=None):
        """
        :param party_model_id: the model id with party info used to identify the model
        :param model_version: the model version
        :param model_object: the converted model object
        :param service_id: name of the service
        :param config_file_content: the content of a config file that will be used to connect to the
                                    kubernetes cluster.
        :param namespace: the kubernetes namespace this service belongs to.
        :param replace: whether to replace the running service, defaults to False.
        :param skip_create_storage_secret: whether or not to skip setting up MinIO credentials for
                                           KFServing storage-initializer container, defaults to False.
        :param model_storage_type: type of the underlying model storage
                                   defaults to ModelStorageType.MINIO.
        :param model_storage_parameters: a dict containing extra arguments to initialize the
                                         model storage instance, defaults to {}.
                                         see the doc of model storage classes for the available
                                         parameters.
        """
        self.party_model_id = party_model_id
        self.model_version = model_version
        self.model_object = model_object
        self.service_id = service_id
        if model_storage_parameters is None:
            model_storage_parameters = {}
        self.model_storage = get_model_storage(storage_type=model_storage_type,
                                               sub_path=party_model_id + "/" + model_version + "/" + service_id,
                                               **model_storage_parameters)
        self.storage_uri = None
        self.isvc = None
        # this should also set up kubernetes.client config
        config_file = None
        if config_file_content:
            config_file = io.StringIO(config_file_content)
        self.kfserving_client = kfserving.KFServingClient(config_file)
        self.namespace = namespace if namespace else kfserving.utils.get_default_target_namespace()
        self.replace = replace
        self.skip_create_storage_secret = skip_create_storage_secret
        stat_logger.debug("Initialized KFServingDeployer with client config: {}".format(config_file))

    def prepare_model(self):
        """
        Prepare the model to be used by KFServing system.

        Calls into each deployer implementation to serialize the model object
        and uploads the related files to the target model storage.

        :return: the uri to fetch the uploaed/prepared model
        """
        if not self.storage_uri:
            self.storage_uri = self.model_storage.save(self._do_prepare_model())
        stat_logger.info("Prepared model with uri: {}".format(self.storage_uri))
        return self.storage_uri

    def _do_prepare_model(self):
        raise NotImplementedError("_do_prepare_storage method not implemented")

    def deploy(self):
        """
        Deploy a KFServing InferenceService from a model object

        :return: the InferenceService object as a dict
        """
        if self.status() and not self.replace:
            raise RuntimeError("serving service {} already exists".format(self.service_id))

        if self.isvc is None:
            stat_logger.info("Preparing model storage and InferenceService spec...")
            self.prepare_model()
            self.prepare_isvc()
        if self.isvc.metadata.annotations is None:
            self.isvc.metadata.annotations = {}
        # add a different annotation to force replace
        self.isvc.metadata.annotations[ANNOTATION_SERVICE_UUID] = str(uuid.uuid4())
        self.isvc.metadata.annotations[ANNOTATION_FATE_MODEL_ID] = self.party_model_id
        self.isvc.metadata.annotations[ANNOTATION_FATE_MODEL_VERSION] = self.model_version

        if isinstance(self.model_storage, MinIOModelStorage) and not self.skip_create_storage_secret:
            self.prepare_sa_secret()

        if self.status() is None:
            stat_logger.info("Creating InferenceService {}...".format(self.service_id))
            created_isvc = self.kfserving_client.create(self.isvc, namespace=self.namespace)
        else:
            stat_logger.info("Replacing InferenceService {}...".format(self.service_id))
            self.isvc.metadata.resource_version = None
            created_isvc = self.kfserving_client.replace(self.service_id, self.isvc,
                                                         namespace=self.namespace)
        return created_isvc

    def prepare_isvc(self):
        """
        Generate an InferenceService spec to be applied into KFServing

        :return: the spec object
        """
        if self.isvc is None:
            self.isvc = kfserving.V1beta1InferenceService(
                api_version=kfserving.constants.KFSERVING_V1BETA1,
                kind=kfserving.constants.KFSERVING_KIND,
                metadata=client.V1ObjectMeta(name=self.service_id),
                spec=kfserving.V1beta1InferenceServiceSpec(
                    predictor=kfserving.V1beta1PredictorSpec()))
            self._do_prepare_predictor()
            if self.namespace:
                self.isvc.metadata.namespace = self.namespace
        stat_logger.info("InferenceService spec ready")
        stat_logger.debug(self.isvc)
        return self.isvc

    def _do_prepare_predictor(self):
        raise NotImplementedError("_do_prepare_predictor method not implemented")

    def destroy(self):
        """
        Delete the InferenceService
        """
        if self.status() is not None:
            self.kfserving_client.delete(self.service_id, namespace=self.namespace)
            stat_logger.info("InferenceService {} is deleted".format(self.service_id))

    def status(self):
        try:
            return self.kfserving_client.get(self.service_id, namespace=self.namespace)
        except RuntimeError as e:
            if "Reason: Not Found" in str(e):
                return None

    def wait(self, timeout=120):
        """Wait until the service becomes ready

        Internally calls KFServing API to retrieve the status

        :param timeout: seconds to wait
        :return: the InferenceService dict
        """
        return self.kfserving_client.get(self.service_id,
                                         namespace=self.namespace,
                                         watch=True,
                                         timeout_seconds=timeout)

    def prepare_sa_secret(self):
        """
        Prepare the secret to be used by the service account for the KFServing service.

        KFServing needs a service account to find the credential to download files
        from MINIO/S3 storage. It must contain a secret resource with the credential
        embedded. We can prepare one use kubernetes API here.
        """
        secrets = client.CoreV1Api().list_namespaced_secret(self.namespace)
        secret_names = [secret.metadata.name for secret in secrets.items]
        annotations = {
            "serving.kubeflow.org/s3-endpoint": self.model_storage.endpoint,
            "serving.kubeflow.org/s3-usehttps": "1" if self.model_storage.secure else "0"
        }
        secret = client.V1Secret(metadata=client.V1ObjectMeta(name=MINIO_K8S_SECRET_NAME,
                                                              annotations=annotations),
                                 type="Opaque",
                                 string_data={
                                     'AWS_ACCESS_KEY_ID': self.model_storage.access_key,
                                     'AWS_SECRET_ACCESS_KEY': self.model_storage.secret_key
                                 })
        if MINIO_K8S_SECRET_NAME not in secret_names:
            client.CoreV1Api().create_namespaced_secret(self.namespace, secret)
        else:
            client.CoreV1Api().patch_namespaced_secret(MINIO_K8S_SECRET_NAME, self.namespace, secret)

        sa_name = self.isvc.spec.predictor.service_account_name \
            if (self.isvc and
                isinstance(self.isvc, kfserving.V1beta1InferenceService) and
                self.isvc.spec.predictor.service_account_name) \
            else "default"
        creds_utils.set_service_account(self.namespace,
                                        sa_name,
                                        MINIO_K8S_SECRET_NAME)
