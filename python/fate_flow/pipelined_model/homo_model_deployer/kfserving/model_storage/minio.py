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

import os
from urllib.parse import urlparse
from minio import Minio

from fate_flow.settings import stat_logger
from .base import BaseModelStorage


ENV_MINIO_ENDPOINT_URL = "MINIO_ENDPOINT_URL"
ENV_MINIO_ACCESS_KEY_ID = "MINIO_ACCESS_KEY_ID"
ENV_MINIO_SECRET_ACCESS_KEY = "MINIO_SECRET_ACCESS_KEY"
ENV_MINIO_USE_HTTPS = "MINIO_USE_HTTPS"
ENV_MINIO_REGION = "MINIO_REGION"


class MinIOModelStorage(BaseModelStorage):
    """Model storage to upload model object into MinIO

    If not specified by the caller, the following environment
    variables will be used to connect to the server:
    MINIO_ENDPOINT_URL, MINIO_ACCESS_KEY_ID, MINIO_SECRET_ACCESS_KEY
    MINIO_USE_HTTPS, MINIO_REGION.
    """

    def __init__(self,
                 sub_path="",
                 bucket="fate-models",
                 endpoint=None,
                 access_key=None,
                 secret_key=None,
                 region=None,
                 secure=True):
        """
        :param sub_path: sub path within the bucket, defaults to ""
        :param bucket: bucket to store the model, defaults to "fate_models"
        :param endpoint: server endpoint, defaults to None
        :param access_key: access key, defaults to None
        :param secret_key: secret key, defaults to None
        :param region: region name, defaults to None
        :param secure: flag to indicate whether tls should be used, defaults to True
        """
        super(MinIOModelStorage, self).__init__()
        self.bucket = bucket
        self.sub_path = sub_path

        if not endpoint:
            url = urlparse(os.getenv(ENV_MINIO_ENDPOINT_URL, "http://minio:9000"))
            secure = url.scheme == 'https' if url.scheme else bool(os.getenv(ENV_MINIO_USE_HTTPS, "true"))
            endpoint = url.netloc
            access_key = os.getenv(ENV_MINIO_ACCESS_KEY_ID)
            secret_key = os.getenv(ENV_MINIO_SECRET_ACCESS_KEY)
            region = os.getenv(ENV_MINIO_REGION)

        stat_logger.debug("Initialized MinIOModelStorage with endpoint: {}, "
                          "access_key: {}, region: {}, with TLS: {}"
                          .format(endpoint, access_key, region, secure))
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.secure = secure

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
            secure=secure
        )

    def save(self, model_object, dest=""):
        """
        Upload the model objects to the specified destination path

        If the dest parameter is empty, the target file/object name
        will be inferred from the local files.

        :param model_object: pointer to the file(s) to be uploaded
        :param dest: the destination file/object name
        :return:
        """
        stat_logger.debug("Upload model object: {} of type: {}"
                          .format(model_object, type(model_object)))
        # If the model object is already a local file then we
        # just upload it.
        if isinstance(model_object, str):
            if os.path.isfile(model_object):
                return self.upload_file(model_object, dest)
            elif os.path.isdir(model_object):
                return self.upload_folder(model_object, dest)
            else:
                raise ValueError("expect an existing path, got: {}".format(model_object))
        elif isinstance(model_object, list):
            for obj in model_object:
                if 'dest' not in obj:
                    obj['dest'] = os.path.basename(obj['file'])
            return self.upload_objects(model_object)
        elif isinstance(model_object, dict) and "dest" in model_object \
                and "file" in model_object and isinstance(model_object['file'], str):
            return self.upload_objects([model_object])
        else:
            raise ValueError("unsupported object type {}".format(type(model_object)))

    def upload_folder(self, folder, dest=""):
        files_to_upload = []
        for dir_, _, files in os.walk(folder):
            for file_name in files:
                rel_dir = os.path.relpath(dir_, folder).lstrip("./")
                rel_dest = os.path.join(rel_dir, file_name)
                if dest:
                    rel_dest = dest + "/" + rel_dest
                file = os.path.join(dir_, file_name)
                files_to_upload.append({"dest": rel_dest, "obj": file})
        return self.upload_objects(files_to_upload)

    def upload_file(self, file, dest=""):
        if not dest:
            dest = os.path.basename(file)
        return self.upload_objects([{"dest": dest, "obj": file}])

    def upload_readable_obj(self, dest, obj, length):
        return self.upload_objects([{"dest": dest,
                                     "obj": obj,
                                     "length": length
                                     }])

    def upload_objects(self, objects):
        """
        Perform the uploading.

        Each element in the objects list should be a dict that looks:

        {
            "obj": <path to a local file or a file-like object>
            "dest": <target file name under "subpath">
            "length": <for a file-like object, length in bytes of the file-like object content>
        }

        :param objects: a list of object to be uploaded.
        :return: the complete "s3://" type uri for the sub-path to local all the uploaded objects
        """
        client = self.client
        bucket = self.bucket
        found = client.bucket_exists(bucket)
        if not found:
            client.make_bucket(bucket)

        for obj in objects:
            dest = self.sub_path + "/" + obj['dest']
            file = obj['obj']
            stat_logger.debug("Uploading {} to {}".format(file, dest))
            if hasattr(file, 'read'):
                length = obj['length']
                client.put_object(bucket, dest, file, length)
            else:
                client.fput_object(bucket, dest, file)
        model_path = f's3://{bucket}/{self.sub_path}'
        stat_logger.info("Uploaded model objects into path: {}".format(model_path))
        return model_path
