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

from arch.api.utils import log_utils

from fate_flow.entity.metric import Metric, MetricMeta
from federatedml.param.union_param import UnionParam
from federatedml.model_base import ModelBase
from federatedml.statistic import data_overview
from federatedml.util.data_io import make_schema

import numpy as np

LOGGER = log_utils.getLogger()


class Union(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = UnionParam()
        self.metric_name = "union"
        self.metric_namespace = "train"
        self.metric_type = "UNION"

    def _init_model(self, params):
        self.model_param = params
        self.need_run = params.need_run
        self.union_func = params.union_func
        self.allow_missing = params.allow_missing
        self.func = None
        self.feature_count = 0

    def _run_data(self, data_sets=None, stage=None):
        if not self.need_run:
            return
        data = {}
        for data_key in data_sets:
            for key in data_sets[data_key].keys():
                if data_sets[data_key].get(key, None):
                    data[data_key] = data_sets[data_key][key]
        if stage == "fit":
            self.data_output = self.fit(data)
        else:
            LOGGER.warning("Union has not transform, return")

    def select_func(self, func_name):
        if func_name == "first":
            return self._keep_first
        if func_name == "last":
            return self._keep_last
        if func_name == "all":
            return self._keep_all
        else:
            raise ValueError("wrong function name given: {}, must be one of 'first', 'last', or 'all'".format(func_name))

    @staticmethod
    def _keep_first(v1, v2):
        return v1

    @staticmethod
    def _keep_last(v1, v2):
        return v2

    @staticmethod
    def _keep_all(v1, v2):
        v1.features = np.hstack((v1.features, v2.features))
        LOGGER.debug("stacked features: {}".format(v1.features))
        if v2.label is not None:
            if v1.label is not None:
                if v1.label != v2.label:
                    raise ValueError("Union try to combine different label value. Union aborted.")
            v1.label = v2.label
        return v1

    def check_schema_id(self, local_schema, old_schema):
        if local_schema.get("sid") != old_schema.get("sid"):
            raise ValueError("Id name does not fit! Check id column names.")

    def check_schema_label_name(self, local_schema, old_schema):
        local_label_name = local_schema.get("label_name")
        old_label_name = old_schema.get("label_name")
        if local_label_name is not None:
            if old_label_name is not None:
                if local_label_name != old_label_name:
                    raise ValueError("Union try to combine tables with different label names. "
                                     "Please check label column names")

    def check_feature_length(self, data_instance):
        if len(data_instance.features) != self.feature_count:
            raise ValueError("Feature length {} mismatch with header length {}.".format(len(data_instance.features), self.feature_count))


    def get_new_label_name(self, local_schema, old_schema):
        local_label_name = local_schema.get("label_name")
        old_label_name = old_schema.get("label_name")
        if local_label_name is not None:
            return local_label_name
        else:
            return old_label_name

    def get_new_schema(self, local_schema, combined_schema):
        if self.union_func == "first":
            new_schema = combined_schema
        elif self.union_func == "last":
            new_schema = local_schema
        elif self.union_func == "all":
            new_header = combined_schema.get("header") + local_schema.get("header")
            new_label_name = self.get_new_label_name(local_schema, combined_schema)
            new_schema = make_schema(header=new_header,
                                     sid_name=combined_schema.get("sid_name"),
                                     label_name=new_label_name)
        else:
            raise ValueError("Illegal union_func received: {}".format(self.union_func))
        self.feature_count = len(new_schema.get("header"))
        return new_schema

    def fit(self, data):
        if len(data) <= 0:
            LOGGER.warning("Union receives no data input.")
            return
        self.func = self.select_func(self.union_func)

        empty_count = 0
        combined_table = None
        combined_schema = None
        metrics = []

        for (key, local_table) in data.items():
            LOGGER.debug("table to combine name {}".format(key))
            num_data = local_table.count()
            local_schema = local_table.schema
            metrics.append(Metric(key, num_data))

            if num_data == 0:
                LOGGER.warning("DTable {} is empty.".format(key))
                empty_count += 1
                continue

            is_empty_feature = data_overview.is_empty_feature(local_table)
            if is_empty_feature:
                LOGGER.warning("DTable {} has no features.".format(key))
                continue

            if combined_table is None:
                # first table to combine
                combined_table = local_table
                combined_schema = local_schema
            else:
                self.check_schema_id(local_schema, combined_schema)
                self.check_schema_label_name(local_schema, combined_schema)
                combined_table = combined_table.union(local_table, self.func)
                combined_table.schema = self.get_new_schema(local_schema, combined_schema)
                combined_schema = combined_table.schema

        if combined_table is None:
            num_data = 0
            LOGGER.warning("All tables provided are empty or have empty features.")
        else:
            if not self.allow_missing:
                combined_table.mapValues(self.check_feature_length)
            num_data = combined_table.count()
        metrics.append(Metric("Total", num_data))

        self.callback_metric(metric_name=self.metric_name,
                             metric_namespace=self.metric_namespace,
                             metric_data=metrics)
        self.tracker.set_metric_meta(metric_namespace=self.metric_namespace,
                                     metric_name=self.metric_name,
                                     metric_meta=MetricMeta(name=self.metric_name, metric_type=self.metric_type))

        LOGGER.debug("after union schema: {}".format(combined_table.schema))

        LOGGER.info("Union operation finished. Total {} empty tables encountered.".format(empty_count))
        return combined_table
