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
import datetime
import json

from peewee import Model, TextField, CompositeKey
from fate_arch.common.base_utils import current_timestamp, serialize_b64, deserialize_b64


class LongTextField(TextField):
    field_type = 'LONGTEXT'


class JSONField(TextField):
    def db_value(self, value):
        if value is None:
            value = {}
        return json.dumps(value)

    def python_value(self, value):
        if value is None:
            value = {}
        return json.loads(value)


class SerializedField(LongTextField):
    def db_value(self, value):
        return serialize_b64(value, to_str=True)

    def python_value(self, value):
        return deserialize_b64(value)


class BaseModel(Model):
    def to_json(self):
        return self.__dict__['__data__']

    def to_human_model_dict(self, only_primary_with: list = None):
        model_dict = self.__dict__["__data__"]
        human_model_dict = {}
        if not only_primary_with:
            for k, v in model_dict.items():
                human_model_dict[k.lstrip("f").lstrip("_")] = v
        else:
            for k in self._meta.primary_key.field_names:
                human_model_dict[k.lstrip("f").lstrip("_")] = model_dict[k]
            for k in only_primary_with:
                human_model_dict[k] = model_dict["f_%s" % k]
        return human_model_dict

    @classmethod
    def get_primary_keys_name(cls):
        return cls._meta.primary_key.field_names if isinstance(cls._meta.primary_key, CompositeKey) else [cls._meta.primary_key.name]

    def save(self, *args, **kwargs):
        if hasattr(self, "f_update_date"):
            self.f_update_date = datetime.datetime.now()
        if hasattr(self, "f_update_time"):
            self.f_update_time = current_timestamp()
        return super(BaseModel, self).save(*args, **kwargs)



