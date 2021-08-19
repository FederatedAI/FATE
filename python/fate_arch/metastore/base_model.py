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
import operator
import typing

from peewee import Field, IntegerField, FloatField, BigIntegerField, TextField, Model, CompositeKey, Metadata
from fate_arch.common.base_utils import current_timestamp, serialize_b64, deserialize_b64, timestamp_to_date, date_string_to_timestamp, json_dumps, json_loads
from fate_arch.common.conf_utils import get_base_config
from fate_arch.common import WorkMode


WORK_MODE = get_base_config('work_mode', 0)
if WORK_MODE == WorkMode.STANDALONE:
    from playhouse.apsw_ext import DateTimeField
elif WORK_MODE == WorkMode.CLUSTER:
    from peewee import DateTimeField
else:
    raise Exception("can not import sql field")

CONTINUOUS_FIELD_TYPE = {IntegerField, FloatField, DateTimeField}
AUTO_DATE_TIMESTAMP_FIELD_PREFIX = {"create", "start", "end", "update", "read_access", "write_access"}


class LongTextField(TextField):
    field_type = 'LONGTEXT'


class JSONField(LongTextField):
    def db_value(self, value):
        if value is None:
            value = {}
        return json_dumps(value)

    def python_value(self, value):
        if value is None:
            return {}
        return json_loads(value)


class SerializedField(LongTextField):
    def db_value(self, value):
        return serialize_b64(value, to_str=True)

    def python_value(self, value):
        return deserialize_b64(value)


def is_continuous_field(cls: typing.Type) -> bool:
    if cls in CONTINUOUS_FIELD_TYPE:
        return True
    for p in cls.__bases__:
        if p in CONTINUOUS_FIELD_TYPE:
            return True
        elif p != Field and p != object:
            if is_continuous_field(p):
                return True
    else:
        return False


def auto_date_timestamp_field():
    return {f"{f}_time" for f in AUTO_DATE_TIMESTAMP_FIELD_PREFIX}


class BaseModel(Model):
    f_create_time = BigIntegerField(null=True)
    f_create_date = DateTimeField(null=True)
    f_update_time = BigIntegerField(null=True)
    f_update_date = DateTimeField(null=True)

    def to_json(self):
        #todo: rename to "to_dict"?
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

    @property
    def meta(self) -> Metadata:
        return self._meta

    @classmethod
    def get_primary_keys_name(cls):
        return cls._meta.primary_key.field_names if isinstance(cls._meta.primary_key, CompositeKey) else [cls._meta.primary_key.name]

    @classmethod
    def getter_by(cls, attr):
        return operator.attrgetter(attr)(cls)

    @classmethod
    def query(cls, reverse=None, order_by=None, **kwargs):
        filters = []
        for f_n, f_v in kwargs.items():
            attr_name = 'f_%s' % f_n
            if not hasattr(cls, attr_name):
                continue
            if type(f_v) in {list, set}:
                f_v = list(f_v)
                if is_continuous_field(type(getattr(cls, attr_name))):
                    if len(f_v) == 2:
                        for i, v in enumerate(f_v):
                            if isinstance(v, str) and f_n in auto_date_timestamp_field():
                                # time type: %Y-%m-%d %H:%M:%S
                                f_v[i] = date_string_to_timestamp(v)
                        lt_value = f_v[0]
                        gt_value = f_v[1]
                        if lt_value is not None and gt_value is not None:
                            filters.append(cls.getter_by(attr_name).between(lt_value, gt_value))
                        elif lt_value is not None:
                            filters.append(operator.attrgetter(attr_name)(cls) >= lt_value)
                        elif gt_value is not None:
                            filters.append(operator.attrgetter(attr_name)(cls) <= gt_value)
                else:
                    filters.append(operator.attrgetter(attr_name)(cls) << f_v)
            else:
                filters.append(operator.attrgetter(attr_name)(cls) == f_v)
        if filters:
            query_records = cls.select().where(*filters)
            if reverse is not None:
                if not order_by or not hasattr(cls, f"f_{order_by}"):
                    order_by = "create_time"
                if reverse is True:
                    query_records = query_records.order_by(cls.getter_by(f"f_{order_by}").desc())
                elif reverse is False:
                    query_records = query_records.order_by(cls.getter_by(f"f_{order_by}").asc())
            return [query_record for query_record in query_records]
        else:
            return []

    def save(self, *args, **kwargs):
        self.f_update_time = current_timestamp()
        for f_n in AUTO_DATE_TIMESTAMP_FIELD_PREFIX:
            if getattr(self, f"f_{f_n}_time", None) and hasattr(self, f"f_{f_n}_date"):
                setattr(self, f"f_{f_n}_date", timestamp_to_date(getattr(self, f"f_{f_n}_time")))
        return super(BaseModel, self).save(*args, **kwargs)

    @classmethod
    def update(cls, __data=None, **update):
        if hasattr(cls, "f_update_time"):
            __data[operator.attrgetter("f_update_time")(cls)] = current_timestamp()
        fields = AUTO_DATE_TIMESTAMP_FIELD_PREFIX.copy()
        # create can not be updated
        fields.remove("create")
        for f_n in fields:
            if hasattr(cls, f"f_{f_n}_time") and hasattr(cls, f"f_{f_n}_date"):
                k = operator.attrgetter(f"f_{f_n}_time")(cls)
                if k in __data and __data[k]:
                    __data[operator.attrgetter(f"f_{f_n}_date")(cls)] = timestamp_to_date(__data[k])
        return super().update(__data, **update)
