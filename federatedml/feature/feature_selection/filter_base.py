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

import operator
import random

from arch.api.utils import log_utils
from federatedml.feature.feature_selection.selection_properties import SelectionProperties

LOGGER = log_utils.getLogger()


class BaseFilterMethod(object):
    """
    Use for filter columns

    Attributes
    ----------
    cols: list of int
        Col index to do selection

    left_cols: dict,
        k is col_index, value is bool that indicate whether it is left or not.

    feature_values: dict
        k is col_name, v is the value that used to judge whether left or not.
    """

    def __init__(self, filter_param):
        self.selection_properties: SelectionProperties = None
        self._parse_filter_param(filter_param)

    @property
    def feature_values(self):
        return self.selection_properties.feature_values

    def fit(self, data_instances, suffix):
        """
        Filter data_instances for the specified columns

        Parameters
        ----------
        data_instances : DTable,
            Input data

        suffix : string,
            Use for transfer_variables

        Returns
        -------
        A list of index of columns left.

        """
        raise NotImplementedError("Should not call this function directly")

    def _parse_filter_param(self, filter_param):
        raise NotImplementedError("Should not call this function directly")

    def set_selection_properties(self, selection_properties):
        self.selection_properties = selection_properties

    def _keep_one_feature(self, pick_high=True):
        """
        Make sure at least one feature can be left after filtering.

        Parameters
        ----------
        pick_high: bool
            Set when none of value left, choose the highest one or lowest one. True means highest one while
            False means lowest one.
        """
        if len(self.selection_properties.left_col_indexes) > 0:
            return

        LOGGER.info("All features has been filtered, keep one without satisfying all the conditions")

        LOGGER.debug("feature values: {}, select_col_names: {}, left_col_names: {}".format(
            self.feature_values, self.selection_properties.select_col_names, self.selection_properties.left_col_names
        ))

        # random pick one
        if len(self.feature_values) == 0:
            left_col_name = random.choice(self.selection_properties.select_col_names)
        else:
            result = sorted(self.feature_values.items(), key=operator.itemgetter(1), reverse=pick_high)
            left_col_name = result[0][0]
        # LOGGER.debug("feature values: {}, left_col_name: {}".format(self.feature_values, left_col_name))

        self.selection_properties.add_left_col_name(left_col_name)

    def set_statics_obj(self, statics_obj):
        # Re-write if needed
        pass

    def set_transfer_variable(self, transfer_variable):
        # Re-write if needed
        pass

    def set_binning_obj(self, binning_model):
        # Re-write if needed
        pass

    def get_meta_obj(self, meta_dicts):
        raise NotImplementedError("Should not call this function directly")
