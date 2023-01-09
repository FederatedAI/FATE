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
import pydantic


class Parameter:
    def parse(self, obj):
        raise NotImplementedError()

    def dict(self):
        raise NotImplementedError()


class ConInt(Parameter):
    def __init__(self, gt: int = None, ge: int = None, lt: int = None, le: int = None) -> None:
        self.gt = gt
        self.ge = ge
        self.lt = lt
        self.le = le

    def parse(self, obj):
        return pydantic.parse_obj_as(pydantic.conint(gt=self.gt, ge=self.ge, lt=self.lt, le=self.le), obj)

    def dict(self):
        meta = {}
        if self.gt is not None:
            meta["gt"] = self.gt
        if self.ge is not None:
            meta["ge"] = self.ge
        if self.lt is not None:
            meta["lt"] = self.lt
        if self.le is not None:
            meta["le"] = self.le
        return meta


class ConFloat(Parameter):
    def __init__(self, gt: float = None, ge: float = None, lt: float = None, le: float = None) -> None:
        self.gt = gt
        self.ge = ge
        self.lt = lt
        self.le = le

    def parse(self, obj):
        return pydantic.parse_obj_as(pydantic.confloat(gt=self.gt, ge=self.ge, lt=self.lt, le=self.le), obj)

    def dict(self):
        meta = {}
        if self.gt is not None:
            meta["gt"] = self.gt
        if self.ge is not None:
            meta["ge"] = self.ge
        if self.lt is not None:
            meta["lt"] = self.lt
        if self.le is not None:
            meta["le"] = self.le
        return meta


def parse(parameter_type, obj):
    if isinstance(parameter_type, Parameter):
        return parameter_type.parse(obj)
    else:
        return pydantic.parse_obj_as(parameter_type, obj)
