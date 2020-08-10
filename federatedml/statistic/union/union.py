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
from arch.api import session

from fate_flow.entity.metric import Metric, MetricMeta
from federatedml.feature.instance import Instance
from federatedml.param.union_param import UnionParam
from federatedml.model_base import ModelBase
from federatedml.statistic import data_overview

LOGGER = log_utils.getLogger()


class Union(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = UnionParam()
        self.metric_name = "union"
        self.metric_namespace = "train"
        self.metric_type = "UNION"
        self.repeated_ids = None
        self.key = None

    def _init_model(self, params):
        self.model_param = params
        self.allow_missing = params.allow_missing
        self.keep_duplicate = params.keep_duplicate
        self.feature_count = 0
        self.is_data_instance = False
        self.is_empty_feature = False

    @staticmethod
    def _keep_first(v1, v2):
        return v1

    def _renew_id(self, k, v):
        result = []
        if k in self.repeated_ids:
            new_k = f"{k}_{self.key}"
            result.append((new_k, v))
        else:
            result.append((k, v))
        return result

    def check_id(self, local_table, combined_table):
        if not self.is_data_instance:
            local_sid_name = local_table.get_meta("sid")
            combined_sid_name = combined_table.get_meta("sid")
        else:
            local_schema, combined_schema = local_table.schema, combined_table.schema
            local_sid_name = local_schema.get("sid")
            combined_sid_name = combined_schema.get("sid")
        if local_sid_name != combined_sid_name:
            raise ValueError(f"Id names {local_sid_name} and {combined_sid_name} do not match! Please check id column names.")

    def check_label_name(self, local_table, combined_table):
        if not self.is_data_instance:
            return
        local_schema, combined_schema = local_table.schema, combined_table.schema
        local_label_name = local_schema.get("label_name")
        combined_label_name = combined_schema.get("label_name")
        if local_label_name is None and combined_label_name is None:
            return
        if local_label_name is None or combined_label_name is None:
            raise ValueError("Union try to combine a labeled data set with an unlabelled one."
                             "Please check labels.")
        if local_label_name != combined_label_name:
            raise ValueError("Label names do not match. "
                                 "Please check label column names.")

    def check_header(self, local_table, combined_table):
        if not self.is_data_instance:
            local_header = local_table.get_meta("header")
            combined_header = combined_table.get_meta("header")
        else:
            local_schema, combined_schema = local_table.schema, combined_table.schema
            local_header = local_schema.get("header")
            combined_header = combined_schema.get("header")
        if local_header != combined_header:
            raise ValueError("Table headers do not match! Please check header.")

    def check_feature_length(self, data_instance):
        if not self.is_data_instance or self.allow_missing:
            return
        if len(data_instance.features) != self.feature_count:
            raise ValueError("Feature length {} mismatch with header length {}.".format(len(data_instance.features), self.feature_count))

    def check_is_data_instance(self, table):
        entry = table.first()
        self.is_data_instance = isinstance(entry[1], Instance)

    def fit(self, data):
        LOGGER.debug(f"fit receives data is {data}")
        if not isinstance(data, dict):
            raise ValueError("Union module must receive more than one table as input.")
        empty_count = 0
        combined_table = None
        combined_schema = None
        metrics = []

        for (key, local_table) in data.items():
            LOGGER.debug("table to combine name: {}".format(key))
            num_data = local_table.count()
            LOGGER.debug("table count: {}".format(num_data))
            metrics.append(Metric(key, num_data))

            if num_data == 0:
                LOGGER.warning("Table {} is empty.".format(key))
                empty_count += 1
                continue
            if combined_table is None:
                self.check_is_data_instance(local_table)
            if self.is_data_instance:
                self.is_empty_feature = data_overview.is_empty_feature(local_table)
                if self.is_empty_feature:
                    LOGGER.warning("Table {} has empty feature.".format(key))

            if combined_table is None:
                # first table to combine
                combined_table = local_table
                if self.is_data_instance:
                    combined_schema = combined_table.schema
                else:
                    combined_metas = combined_table.get_metas()

            else:
                self.check_id(local_table, combined_table)
                self.check_label_name(local_table, combined_table)
                self.check_header(local_table, combined_table)
                if self.keep_duplicate:
                    repeated_ids = combined_table.join(local_table, lambda v1, v2: 1)
                    self.repeated_ids = [v[0] for v in repeated_ids.collect()]
                    self.key = key
                    local_table = local_table.flatMap(self._renew_id)

                combined_table = combined_table.union(local_table, self._keep_first)

                if self.is_data_instance:
                    combined_table.schema = combined_schema
                else:
                    combined_metas["namespace"] = combined_table.get_namespace()
                    session.save_data_table_meta(combined_metas, combined_table.get_name(),
                                                 combined_table.get_namespace())

            # only check feature length if not empty
            if self.is_data_instance and not self.is_empty_feature:
                self.feature_count = len(combined_schema.get("header"))
                LOGGER.debug("feature count: {}".format(self.feature_count))
                combined_table.mapValues(self.check_feature_length)


        if combined_table is None:
            num_data = 0
            LOGGER.warning("All tables provided are empty or have empty features.")
        else:
            num_data = combined_table.count()
        metrics.append(Metric("Total", num_data))

        self.callback_metric(metric_name=self.metric_name,
                             metric_namespace=self.metric_namespace,
                             metric_data=metrics)
        self.tracker.set_metric_meta(metric_namespace=self.metric_namespace,
                                     metric_name=self.metric_name,
                                     metric_meta=MetricMeta(name=self.metric_name, metric_type=self.metric_type))

        LOGGER.info("Union operation finished. Total {} empty tables encountered.".format(empty_count))
        # if not self.is_data_instance:
        #   LOGGER.debug(f"output dtable's metas is {combined_table.get_metas()}")
        return combined_table

    def check_consistency(self):
        pass

