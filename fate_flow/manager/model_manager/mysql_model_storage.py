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
import sys
import datetime
from peewee import Model, CharField, BigIntegerField, TextField, CompositeKey
from playhouse.pool import PooledMySQLDatabase

from fate_flow.manager.model_manager.pipelined_model import PipelinedModel
from fate_flow.manager.model_manager.model_storage_base import ModelStorageBase
from arch.api.utils import log_utils
from arch.api.utils.core_utils import current_timestamp, serialize_b64, deserialize_b64

LOGGER = log_utils.getLogger()
DB = PooledMySQLDatabase(None)


class MysqlModelStorage(ModelStorageBase):
    def __init__(self):
        super(MysqlModelStorage, self).__init__()

    def store(self, model_id: str, model_version: str, store_address: dict, force_update: bool = False):
        """
        Store the model from local cache to mysql
        :param model_id:
        :param model_version:
        :param store_address:
        :param force_update:
        :return:
        """
        try:
            self.get_connection(config=store_address)
            DB.create_tables([MachineLearningModel])
            model_in_table = MachineLearningModel()
            model = PipelinedModel(model_id=model_id, model_version=model_version)
            with DB.connection_context():
                with open(model.packaging_model(), "rb") as fr:
                    LOGGER.info("start store model {} {}".format(model_id, model_version))
                    model_in_table.f_create_time = current_timestamp()
                    model_in_table.f_model_id = model_id
                    model_in_table.f_model_version = model_version
                    model_in_table.f_content = serialize_b64(fr.read())
                    model_in_table.f_size = sys.getsizeof(model_in_table.f_content)
                if force_update:
                    model_in_table.save(only=[MachineLearningModel.f_content, MachineLearningModel.f_size, MachineLearningModel.f_update_time])
                    LOGGER.info("update model {} {} content".format(model_id, model_version))
                else:
                    model_in_table.save(force_insert=True)
                    LOGGER.info("insert model {} {} content".format(model_id, model_version))
            LOGGER.info("Store model {} {} to mysql successfully".format(model_id,
                                                                         model_version))
            self.close_connection()
        except Exception as e:
            LOGGER.exception(e)
            raise Exception("Store model {} {} to mysql failed".format(model_id, model_version))

    def restore(self, model_id: str, model_version: str, store_address: dict):
        """
        Restore model from mysql to local cache
        :param model_id:
        :param model_version:
        :param store_address:
        :return:
        """
        try:
            self.get_connection(config=store_address)
            model = PipelinedModel(model_id=model_id, model_version=model_version)
            with DB.connection_context():
                models_in_table = MachineLearningModel.select().where(MachineLearningModel.f_model_id==model_id, MachineLearningModel.f_model_version==model_version)
                if not models_in_table:
                    raise Exception("Restore model {} {} from mysql failed: {}".format(
                        model_id, model_version, "can not found model in table"))
                model_in_table = models_in_table[0]
                model_archive_data = deserialize_b64(model_in_table.f_content)
                if not model_archive_data:
                    raise Exception("Restore model {} {} from mysql failed: {}".format(
                        model_id, model_version, "can not get model archive data"))
                with open(model.archive_model_file_path(), "wb") as fw:
                    fw.write(model_archive_data)
                model.unpack_model(model.archive_model_file_path())
                LOGGER.info("Restore model to {} from mysql successfully".format(model.archive_model_file_path()))
            self.close_connection()
        except Exception as e:
            LOGGER.exception(e)
            raise Exception("Restore model {} {} from mysql failed".format(model_id, model_version))

    def get_connection(self, config: dict):
        db_name = config["name"]
        config.pop("name")
        DB.init(db_name, **config)

    def close_connection(self):
        try:
            if DB:
                DB.close()
        except Exception as e:
            LOGGER.exception(e)

    def store_key(self, model_id: str, model_version: str):
        return ":".join(["FATEFlow", "PipelinedModel", model_id, model_version])


class DataBaseModel(Model):
    class Meta:
        database = DB

    def to_json(self):
        return self.__dict__['__data__']

    def save(self, *args, **kwargs):
        if hasattr(self, "f_update_date"):
            self.f_update_date = datetime.datetime.now()
        if hasattr(self, "f_update_time"):
            self.f_update_time = current_timestamp()
        super(DataBaseModel, self).save(*args, **kwargs)


class MachineLearningModel(DataBaseModel):
    f_model_id = CharField(max_length=100, index=True)
    f_model_version = CharField(max_length=100, index=True)
    f_size = BigIntegerField(default=0)
    f_create_time = BigIntegerField(default=0)
    f_update_time = BigIntegerField(default=0)
    f_description = TextField(null=True, default='')
    f_content = TextField(default='')

    class Meta:
        db_table = "t_machine_learning_model"
        primary_key = CompositeKey('f_model_id', 'f_model_version')
