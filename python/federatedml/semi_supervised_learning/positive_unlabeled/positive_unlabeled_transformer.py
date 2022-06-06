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

from fate_arch.session import computing_session
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.model_base import ModelBase
from federatedml.model_base import Metric
from federatedml.feature.instance import Instance
from federatedml.param.positive_unlabeled_param import PositiveUnlabeledParam


class PositiveUnlabeled(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = PositiveUnlabeledParam()
        self.metric_name = "positive_unlabeled"
        self.metric_namespace = "train"

    def _init_model(self, model_param):
        self.reverse_order = model_param.reverse_order
        self.threshold_percent = model_param.threshold_percent
        self.pu_mode = model_param.pu_mode

        if self.reverse_order:
            self.replaced_value = 1
        else:
            self.replaced_value = 0

    def to_instance(self, features, label):
        return Instance(features=features, label=label)

    def to_data_inst(self, feature_table, label_table):
        return feature_table.join(label_table, lambda f, l: self.to_instance(f, l))

    def fit(self, data_insts):
        LOGGER.info("Convert label by positive unlabeled transformer")
        data_insts_list = list(data_insts.values())

        if self.role == consts.GUEST:
            LOGGER.info("Identify intersect and predict table")
            if "predict_score" not in data_insts_list[0].schema["header"]:
                intersect_table, predict_table = data_insts_list[0], data_insts_list[1]
            else:
                intersect_table, predict_table = data_insts_list[1], data_insts_list[0]

            LOGGER.info("Extract tables of feature, label and predict score")
            feature_table = intersect_table.mapValues(lambda x: x.features)
            label_score_table = predict_table.mapValues(lambda x: x.features[0:3:2])

            LOGGER.info("Compute threshold index")
            label_score_num = label_score_table.count()
            threshold_idx = int(label_score_num * self.threshold_percent)

            LOGGER.info("Sort based on predict score")
            label_score_list = list(label_score_table.collect())
            label_score_list.sort(key=lambda x: x[1][1], reverse=self.reverse_order)

            LOGGER.info("Replace label based on threshold index")
            replaced_label_list = []
            if self.pu_mode == "two_step" or self.pu_mode == "two_step":
                LOGGER.info("Execute two-step mode")
                unlabeled_to_positive_count, unlabeled_to_negative_count = 0, 0
                for idx, (k, v) in enumerate(label_score_list):
                    if idx < threshold_idx and v[0] == -1:
                        replaced_label_list.append((k, self.replaced_value))
                        unlabeled_to_positive_count += 1
                    elif idx > (label_score_num - threshold_idx) and v[0] == -1:
                        replaced_label_list.append((k, 1 - self.replaced_value))
                        unlabeled_to_negative_count += 1
                    else:
                        replaced_label_list.append((k, int(v[0])))

                self.add_summary("count of unlabeled to positive", unlabeled_to_positive_count)
                self.add_summary("count of unlabeled to negative", unlabeled_to_negative_count)
                self.callback_metric(metric_name=self.metric_name,
                                     metric_namespace=self.metric_namespace,
                                     metric_data=[Metric("count of unlabeled to positive",
                                                         unlabeled_to_positive_count),
                                                  Metric("count of unlabeled to negative",
                                                         unlabeled_to_negative_count)])
            else:
                LOGGER.info("Execute standard mode")
                converted_unlabeled_count = 0
                for idx, (k, v) in enumerate(label_score_list):
                    if idx < threshold_idx and v[0] == 0:
                        replaced_label_list.append((k, self.replaced_value))
                        converted_unlabeled_count += 1
                    else:
                        replaced_label_list.append((k, int(v[0])))

                self.add_summary("count of converted unlabeled", converted_unlabeled_count)
                self.callback_metric(metric_name=self.metric_name,
                                     metric_namespace=self.metric_namespace,
                                     metric_data=[Metric("count of converted unlabeled", converted_unlabeled_count)])

            LOGGER.info("Construct replaced label table")
            replaced_label_table = computing_session.parallelize(replaced_label_list,
                                                                 include_key=True,
                                                                 partition=intersect_table.partitions)
            replaced_label_feature_table = self.to_data_inst(feature_table, replaced_label_table)
            replaced_label_feature_table.schema = intersect_table.schema
            return replaced_label_feature_table

        elif self.role == consts.HOST:
            LOGGER.info("Identify intersect table")
            if data_insts_list[0]:
                intersect_table = data_insts_list[0]
            else:
                intersect_table = data_insts_list[1]
            return intersect_table
