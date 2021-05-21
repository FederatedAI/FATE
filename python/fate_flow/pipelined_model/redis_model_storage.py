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
import redis

from fate_flow.pipelined_model.pipelined_model import PipelinedModel
from fate_flow.pipelined_model.model_storage_base import ModelStorageBase
from fate_arch.common import log

LOGGER = log.getLogger()


class RedisModelStorage(ModelStorageBase):
    def __init__(self):
        super(RedisModelStorage, self).__init__()

    def store(self, model_id: str, model_version: str, store_address: dict, force_update: bool = False):
        """
        Store the model from local cache to redis
        :param model_id:
        :param model_version:
        :param store_address:
        :param force_update:
        :return:
        """
        try:
            red = self.get_connection(config=store_address)
            model = PipelinedModel(model_id=model_id, model_version=model_version)
            redis_store_key = self.store_key(model_id=model_id, model_version=model_version)
            with open(model.packaging_model(), "rb") as fr:
                red.set(name=redis_store_key,
                        value=fr.read(),
                        ex=store_address.get("ex", None),
                        nx=True if not force_update else False
                        )
            LOGGER.info("Store model {} {} to redis successfully using key {}".format(model_id,
                                                                                  model_version,
                                                                                  redis_store_key))
        except Exception as e:
            LOGGER.exception(e)
            raise Exception("Store model {} {} to redis failed".format(model_id, model_version))

    def restore(self, model_id: str, model_version: str, store_address: dict):
        """
        Restore model from redis to local cache
        :param model_id:
        :param model_version:
        :param store_address:
        :return:
        """
        try:
            red = self.get_connection(config=store_address)
            model = PipelinedModel(model_id=model_id, model_version=model_version)
            redis_store_key = self.store_key(model_id=model_id, model_version=model_version)
            model_archive_data = red.get(name=redis_store_key)
            if not model_archive_data:
                raise Exception("Restore model {} {} to redis failed: {}".format(
                    model_id, model_version, "can not found model archive data"))
            with open(model.archive_model_file_path(), "wb") as fw:
                fw.write(model_archive_data)
            model.unpack_model(model.archive_model_file_path())
            LOGGER.info("Restore model to {} from redis successfully using key {}".format(model.archive_model_file_path(),
                                                                                          redis_store_key))
        except Exception as e:
            LOGGER.exception(e)
            raise Exception("Restore model {} {} from redis failed".format(model_id, model_version))

    def get_connection(self, config: dict):
        red = redis.Redis(host=config.get("host", None),
                          port=int(config.get("port", 0)),
                          db=int(config.get("db", 0)),
                          password=config.get("password", None),
                          )
        return red

    def store_key(self, model_id: str, model_version: str):
        return ":".join(["FATEFlow", "PipelinedModel", model_id, model_version])
