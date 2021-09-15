#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import builtins
import json
import os

from federatedml.util import LOGGER, consts

_DEPRECATED_ATTR = "_deprecated_params_set"
_USER_FEEDED_ATTR = "_user_feeded_params"


def deprecated_param(*names):
    def _decorator(cls: "BaseParam"):
        if not cls._has_deprecated_params_set():
            cls._init_deprecated_params_set()

        for name in names:
            cls._set_deprecated_params(name, False)
        return cls

    return _decorator


class BaseParam(object):
    def __init__(self):
        pass

    def set_name(self, name: str):
        self._name = name
        return self

    def check(self):
        raise NotImplementedError("Parameter Object should have be check")

    @classmethod
    def _has_deprecated_params_set(cls):
        return hasattr(cls, _DEPRECATED_ATTR)

    @classmethod
    def _init_deprecated_params_set(cls):
        setattr(cls, _DEPRECATED_ATTR, {})

    @classmethod
    def _get_deprecated_params_set(cls):
        return getattr(cls, _DEPRECATED_ATTR, {})

    @classmethod
    def _set_deprecated_params(cls, name, status):
        cls._get_deprecated_params_set()[name] = status

    @classmethod
    def _get_deprecated_param_status(cls, name, default):
        return cls._get_deprecated_params_set().get(name, default)

    def get_user_feeded(self):
        return getattr(self, _USER_FEEDED_ATTR, [])

    @classmethod
    def get_deprecate_params(cls):
        return getattr(cls, _DEPRECATED_ATTR, {})

    def as_dict(self):
        def _recursive_convert_obj_to_dict(obj):
            ret_dict = {}
            for variable in obj.__dict__:

                # ignore default deprecated params
                if not self._get_deprecated_param_status(variable, True):
                    continue

                attr = getattr(obj, variable)
                if attr and type(attr).__name__ not in dir(builtins):
                    ret_dict[variable] = _recursive_convert_obj_to_dict(attr)
                else:
                    ret_dict[variable] = attr

            return ret_dict

        return _recursive_convert_obj_to_dict(self)

    @classmethod
    def from_dict(cls, conf):
        obj = cls()
        obj.update(conf)
        return obj

    def update(self, conf, allow_redundant=False):
        def _recursive_update_param(param, config, depth):
            if depth > consts.PARAM_MAXDEPTH:
                raise ValueError("Param define nesting too deep!!!, can not parse it")

            inst_variables = param.__dict__
            redundant_attrs = []
            for config_key, config_value in config.items():
                # redundant attr
                if config_key not in inst_variables:
                    # set inner attr
                    if config_key.startswith("_"):
                        setattr(param, config_key, config_value)
                    else:
                        redundant_attrs.append(config_key)
                    continue

                # deprecated param set
                if config_key in self._get_deprecated_params_set():
                    self._set_deprecated_params(config_key, status=True)

                # supported attr
                attr = getattr(param, config_key)
                if type(attr).__name__ in dir(builtins) or attr is None:
                    setattr(param, config_key, config_value)

                else:
                    # recursive set obj attr
                    sub_params = _recursive_update_param(attr, config_value, depth + 1)
                    setattr(param, config_key, sub_params)

            if not allow_redundant and redundant_attrs:
                raise ValueError(
                    f"cpn `{getattr(self, '_name', type(self))}` has redundant parameters: `{[redundant_attrs]}`"
                )

            if getattr(self, _USER_FEEDED_ATTR, None) is None:
                setattr(self, _USER_FEEDED_ATTR, list(conf.keys()))
            return param

        return _recursive_update_param(param=self, config=conf, depth=0)

    def extract_not_builtin(self):
        def _get_not_builtin_types(obj):
            ret_dict = {}
            for variable in obj.__dict__:
                attr = getattr(obj, variable)
                if attr and type(attr).__name__ not in dir(builtins):
                    ret_dict[variable] = _get_not_builtin_types(attr)

            return ret_dict

        return _get_not_builtin_types(self)

    def validate(self):
        self.builtin_types = dir(builtins)
        self.func = {
            "ge": self._greater_equal_than,
            "le": self._less_equal_than,
            "in": self._in,
            "not_in": self._not_in,
            "range": self._range,
        }
        home_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        param_validation_path_prefix = home_dir + "/param_validation/"

        param_name = type(self).__name__
        param_validation_path = "/".join(
            [param_validation_path_prefix, param_name + ".json"]
        )

        validation_json = None
        print("param validation path is {}".format(home_dir))

        try:
            with open(param_validation_path, "r") as fin:
                validation_json = json.loads(fin.read())
        except:
            return

        self._validate_param(self, validation_json)

    def _validate_param(self, param_obj, validation_json):
        default_section = type(param_obj).__name__
        var_list = param_obj.__dict__

        for variable in var_list:
            attr = getattr(param_obj, variable)

            if type(attr).__name__ in self.builtin_types or attr is None:
                if variable not in validation_json:
                    continue

                validation_dict = validation_json[default_section][variable]
                value = getattr(param_obj, variable)
                value_legal = False

                for op_type in validation_dict:
                    if self.func[op_type](value, validation_dict[op_type]):
                        value_legal = True
                        break

                if not value_legal:
                    raise ValueError(
                        "Plase check runtime conf, {} = {} does not match user-parameter restriction".format(
                            variable, value
                        )
                    )

            elif variable in validation_json:
                self._validate_param(attr, validation_json)

    @staticmethod
    def check_string(param, descr):
        if type(param).__name__ not in ["str"]:
            raise ValueError(
                descr + " {} not supported, should be string type".format(param)
            )

    @staticmethod
    def check_positive_integer(param, descr):
        if type(param).__name__ not in ["int", "long"] or param <= 0:
            raise ValueError(
                descr + " {} not supported, should be positive integer".format(param)
            )

    @staticmethod
    def check_positive_number(param, descr):
        if type(param).__name__ not in ["float", "int", "long"] or param <= 0:
            raise ValueError(
                descr + " {} not supported, should be positive numeric".format(param)
            )

    @staticmethod
    def check_nonnegative_number(param, descr):
        if type(param).__name__ not in ["float", "int", "long"] or param < 0:
            raise ValueError(
                descr
                + " {} not supported, should be non-negative numeric".format(param)
            )

    @staticmethod
    def check_decimal_float(param, descr):
        if type(param).__name__ not in ["float", "int"] or param < 0 or param > 1:
            raise ValueError(
                descr
                + " {} not supported, should be a float number in range [0, 1]".format(
                    param
                )
            )

    @staticmethod
    def check_boolean(param, descr):
        if type(param).__name__ != "bool":
            raise ValueError(
                descr + " {} not supported, should be bool type".format(param)
            )

    @staticmethod
    def check_open_unit_interval(param, descr):
        if type(param).__name__ not in ["float"] or param <= 0 or param >= 1:
            raise ValueError(
                descr + " should be a numeric number between 0 and 1 exclusively"
            )

    @staticmethod
    def check_valid_value(param, descr, valid_values):
        if param not in valid_values:
            raise ValueError(
                descr
                + " {} is not supported, it should be in {}".format(param, valid_values)
            )

    @staticmethod
    def check_defined_type(param, descr, types):
        if type(param).__name__ not in types:
            raise ValueError(
                descr + " {} not supported, should be one of {}".format(param, types)
            )

    @staticmethod
    def check_and_change_lower(param, valid_list, descr=""):
        if type(param).__name__ != "str":
            raise ValueError(
                descr
                + " {} not supported, should be one of {}".format(param, valid_list)
            )

        lower_param = param.lower()
        if lower_param in valid_list:
            return lower_param
        else:
            raise ValueError(
                descr
                + " {} not supported, should be one of {}".format(param, valid_list)
            )

    @staticmethod
    def _greater_equal_than(value, limit):
        return value >= limit - consts.FLOAT_ZERO

    @staticmethod
    def _less_equal_than(value, limit):
        return value <= limit + consts.FLOAT_ZERO

    @staticmethod
    def _range(value, ranges):
        in_range = False
        for left_limit, right_limit in ranges:
            if (
                left_limit - consts.FLOAT_ZERO
                <= value
                <= right_limit + consts.FLOAT_ZERO
            ):
                in_range = True
                break

        return in_range

    @staticmethod
    def _in(value, right_value_list):
        return value in right_value_list

    @staticmethod
    def _not_in(value, wrong_value_list):
        return value not in wrong_value_list

    def _warn_deprecated_param(self, param_name, descr):
        if self._deprecated_params_set.get(param_name):
            LOGGER.warning(
                f"{descr} {param_name} is deprecated and ignored in this version."
            )

    def _warn_to_deprecate_param(self, param_name, descr, new_param):
        if self._deprecated_params_set.get(param_name):
            LOGGER.warning(
                f"{descr} {param_name} will be deprecated in future release; "
                f"please use {new_param} instead."
            )
            return True
        return False
