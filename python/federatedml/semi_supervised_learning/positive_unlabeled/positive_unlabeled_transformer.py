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
import copy

from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.model_base import ModelBase
from federatedml.model_base import Metric
from federatedml.param.positive_unlabeled_param import PositiveUnlabeledParam


class PositiveUnlabeled(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = PositiveUnlabeledParam()
        self.metric_name = "positive_unlabeled"
        self.metric_namespace = "train"
        self.replaced_label_list = []
        self.converted_unlabeled_count = 0

    def _init_model(self, model_param):
        self.strategy = model_param.strategy
        self.threshold = model_param.threshold

    def probability_process(self, label_score_table):
        def replaced_func(x):
            if x[1] >= self.threshold and x[0] == 0:
                return 1
            else:
                return x[0]

        def summarized_func(r, l):
            if r == 1 and l[0] == 0:
                return 1
            else:
                return 0

        LOGGER.info("Switch probability strategy")
        replaced_label_table = label_score_table.mapValues(replaced_func)
        self.replaced_label_list = list(replaced_label_table.collect())

        summary_table = replaced_label_table.join(label_score_table, summarized_func)
        self.converted_unlabeled_count = summary_table.filter(lambda k, v: v == 1).count()

    def quantity_process(self, label_score_table):
        LOGGER.info("Switch quantity strategy")
        label_score_list = list(label_score_table.collect())
        label_score_list.sort(key=lambda x: x[1][1], reverse=True)

        LOGGER.info("Count unlabeled samples")
        label_list = [v[0] for (_, v) in label_score_list]
        unlabeled_count = label_list.count(0)
        if int(self.threshold) > unlabeled_count:
            LOGGER.warning("Param 'threshold' should no larger than unlabeled count")

        accumulated_count = 0
        for idx, (k, v) in enumerate(label_score_list):
            if accumulated_count < int(self.threshold) and v[0] == 0:
                self.replaced_label_list.append((k, 1))
                self.converted_unlabeled_count += 1
                accumulated_count += 1
            else:
                self.replaced_label_list.append((k, int(v[0])))

    def proportion_process(self, label_score_table):
        LOGGER.info("Switch proportion strategy")
        label_score_list = list(label_score_table.collect())
        label_score_list.sort(key=lambda x: x[1][1], reverse=True)

        LOGGER.info("Compute threshold index")
        total_num = label_score_table.count()
        threshold_idx = int(total_num * self.threshold)

        for idx, (k, v) in enumerate(label_score_list):
            if idx < threshold_idx and v[0] == 0:
                self.replaced_label_list.append((k, 1))
                self.converted_unlabeled_count += 1
            else:
                self.replaced_label_list.append((k, int(v[0])))

    def distribution_process(self, label_score_table):
        LOGGER.info("Switch distribution strategy")
        label_score_list = list(label_score_table.collect())
        label_score_list.sort(key=lambda x: x[1][1], reverse=True)

        LOGGER.info("Compute threshold index")
        total_num = label_score_table.count()
        unlabeled_num = label_score_table.filter(lambda k, v: v[0] == 0).count()
        threshold_idx = int((unlabeled_num / total_num) * self.threshold)

        for idx, (k, v) in enumerate(label_score_list):
            if idx < threshold_idx and v[0] == 0:
                self.replaced_label_list.append((k, 1))
                self.converted_unlabeled_count += 1
            else:
                self.replaced_label_list.append((k, int(v[0])))

    def apply_labeling_strategy(self, strategy, label_score_table):
        if strategy == consts.PROBABILITY:
            self.probability_process(label_score_table)
        elif strategy == consts.QUANTITY:
            self.quantity_process(label_score_table)
        elif strategy == consts.PROPORTION:
            self.proportion_process(label_score_table)
        else:
            self.distribution_process(label_score_table)

    def callback_info(self):
        self.add_summary("count of converted unlabeled", self.converted_unlabeled_count)
        self.callback_metric(metric_name=self.metric_name,
                             metric_namespace=self.metric_namespace,
                             metric_data=[Metric("count of converted unlabeled", self.converted_unlabeled_count)])

    @staticmethod
    def replace_instance_label(key, val, replaced_label_list):
        replaced_label_dict = dict(replaced_label_list)
        copied_val = copy.deepcopy(val)
        copied_val.label = replaced_label_dict[key]
        return copied_val

    def fit(self, data_insts):
        LOGGER.info("Convert labels by positive unlabeled transformer")
        data_insts_list = list(data_insts.values())

        if self.role == consts.GUEST:
            LOGGER.info("Identify intersect and predict table")
            if "predict_score" not in data_insts_list[0].schema["header"]:
                intersect_table, predict_table = data_insts_list[0], data_insts_list[1]
            else:
                intersect_table, predict_table = data_insts_list[1], data_insts_list[0]

            LOGGER.info("Extract table of label and predict score")
            label_score_table = predict_table.mapValues(lambda x: x.features[0:3:2])

            LOGGER.info("Replace labels by labeling strategy")
            self.apply_labeling_strategy(strategy=self.strategy, label_score_table=label_score_table)

            LOGGER.info("Construct replaced label feature table")
            replaced_label_feature_table = intersect_table.map(
                lambda k, v: (k, self.replace_instance_label(k, v, self.replaced_label_list)))
            replaced_label_feature_table.schema = intersect_table.schema

            LOGGER.info("Obtain positive unlabeled component summary")
            self.callback_info()

            return replaced_label_feature_table

        elif self.role == consts.HOST:
            LOGGER.info("Identify intersect table")
            if data_insts_list[0]:
                intersect_table = data_insts_list[0]
            else:
                intersect_table = data_insts_list[1]

            return intersect_table
