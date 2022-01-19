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

import numpy as np

from federatedml.feature.feature_selection.iso_model_filter import FederatedIsoModelFilter
from federatedml.param.feature_selection_param import CorrelationFilterParam
from federatedml.protobuf.generated import feature_selection_meta_pb2
from federatedml.util.component_properties import ComponentProperties
from federatedml.util import LOGGER


class CorrelationFilter(FederatedIsoModelFilter):
    """
    filter the columns if all values in this feature is the same

    """

    def __init__(self, filter_param: CorrelationFilterParam, external_model, correlation_model,
                 role, cpp: ComponentProperties):
        super().__init__(filter_param, iso_model=external_model, role=role, cpp=cpp)
        self.correlation_model = correlation_model
        self.host_party_id = int(self.correlation_model.parties[1][1:-1].split(",")[1])
        self.take_high = False

    def _parse_filter_param(self, filter_param: CorrelationFilterParam):
        self.sort_metric = filter_param.sort_metric
        self.threshold = filter_param.threshold
        self.select_federated = filter_param.select_federated

    def get_meta_obj(self):
        result = feature_selection_meta_pb2.FilterMeta(
            metrics="correlation",
            filter_type="Sort and filter by threshold",
            threshold=self.threshold,
            select_federated=self.select_federated
        )
        return result

    def _guest_fit(self, suffix):
        sorted_idx, col_names = self.__sort_features()
        filtered_name, host_filtered_name = self.__select_corr(sorted_idx, col_names)
        # LOGGER.debug(f"select_col_name: {self.selection_properties.select_col_names}")
        for name in self.selection_properties.select_col_names:
            if name not in filtered_name:
                self.selection_properties.add_left_col_name(name)
                self.selection_properties.add_feature_value(name, 0.0)
            else:
                self.selection_properties.add_feature_value(name, filtered_name[name])

        if self.select_federated:
            host_id = self.cpp.host_party_idlist.index(self.host_party_id)
            host_prop = self.host_selection_properties[host_id]
            for name in host_prop.select_col_names:
                if name not in host_filtered_name:
                    host_prop.add_left_col_name(name)
                    host_prop.add_feature_value(name, 0.0)
                else:
                    host_prop.add_feature_value(name, host_filtered_name[name])
            self._keep_one_feature(pick_high=self.take_high, selection_properties=host_prop,
                                   feature_values=[])
        if self.select_federated:
            self.sync_obj.sync_select_results(self.host_selection_properties, suffix=suffix)

    def __select_corr(self, sorted_idx, col_names):
        guest_col_names = self.correlation_model.col_names
        host_col_names = self.correlation_model.host_col_names
        filtered_name = {}
        host_filtered_name = {}
        for idx in sorted_idx:
            party, name = col_names[idx]
            if name in filtered_name:
                continue
            if party == 'guest':
                row = guest_col_names.index(name)
                corr = self.correlation_model.local_corr[row, :]
                filtered_name = self.__get_filtered_column(corr, filtered_name, guest_col_names, name, True)
                corr = self.correlation_model.corr[row, :]
                host_filtered_name = self.__get_filtered_column(corr, host_filtered_name,
                                                                host_col_names, name, False)
                # LOGGER.debug(f"guest_col_name: {name}, filtered_name: {filtered_name}, "
                #             f"host_filtered_name: {host_filtered_name}")
            else:
                column = host_col_names.index(name)
                corr = self.correlation_model.corr[:, column]
                filtered_name = self.__get_filtered_column(corr, filtered_name, guest_col_names, name, False)
                # LOGGER.debug(f"host_col_name: {name}, filtered_name: {filtered_name}, "
                #             f"host_filtered_name: {host_filtered_name}")
        return filtered_name, host_filtered_name

    def __get_filtered_column(self, corr, filtered_name, all_names, curt_name, is_local=True):
        for idx, v in enumerate(corr):
            if np.abs(v) > self.threshold:
                _name = all_names[idx]
                if is_local and _name == curt_name:
                    continue
                if _name in filtered_name:
                    continue
                else:
                    filtered_name[_name] = v
        return filtered_name

    def __sort_features(self):
        metric_info = self.iso_model.get_metric_info(self.sort_metric)
        all_feature_values = metric_info.get_partial_values(self.selection_properties.select_col_names)
        col_names = [("guest", x) for x in self.selection_properties.select_col_names]

        if self.select_federated:
            assert len(self.correlation_model.parties) == 2, "Correlation Model should contain host info" \
                                                             "for select_federated in correlation_filter"
            LOGGER.debug(f"correlation_parties: {self.correlation_model.parties}")
            host_id = self.cpp.host_party_idlist.index(self.host_party_id)
            host_property = self.host_selection_properties[host_id]
            all_feature_values.extend(metric_info.get_partial_values(
                host_property.select_col_names, self.host_party_id
            ))
            col_names.extend([(self.host_party_id, x) for x in host_property.select_col_names])
        sorted_idx = np.argsort(all_feature_values)[::-1]
        return sorted_idx, col_names
