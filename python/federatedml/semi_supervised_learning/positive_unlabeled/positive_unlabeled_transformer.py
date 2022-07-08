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
        self.replaced_label_list = []
        self.unlabeled_to_positive_count = 0
        self.unlabeled_to_negative_count = 0
        self.converted_unlabeled_count = 0

    def _init_model(self, model_param):
        self.mode = model_param.mode
        self.labeling_strategy = model_param.labeling_strategy
        self.threshold_percent = model_param.threshold_percent
        self.threshold_amount = model_param.threshold_amount
        self.threshold_proba = model_param.threshold_proba

    def to_instance(self, features, label):
        return Instance(features=features, label=label)

    def to_data_inst(self, feature_table, label_table):
        return feature_table.join(label_table, lambda f, l: self.to_instance(f, l))

    def get_properties(self, target, origin):
        target.inst_id = origin.inst_id
        target.set_weight(origin.weight)
        return target

    def assign_properties(self, target_inst, origin_inst):
        return target_inst.join(origin_inst, lambda t, o: self.get_properties(t, o))

    def proportion_processing(self, label_score_table):
        LOGGER.info("Switch proportion strategy")
        label_score_list = list(label_score_table.collect())
        label_score_list.sort(key=lambda x: x[1][1], reverse=True)

        LOGGER.info("Compute threshold index")
        label_score_num = label_score_table.count()
        threshold_idx = int(label_score_num * self.threshold_percent)

        if self.mode == consts.TWO_STEP:
            LOGGER.info("Execute two-step mode")
            if self.threshold_percent > 0.5:
                LOGGER.warning("Param 'threshold_percent' should no larger than 50% in two-step mode")

            for idx, (k, v) in enumerate(label_score_list):
                if idx < threshold_idx and v[0] == -1:
                    self.replaced_label_list.append((k, 1))
                    self.unlabeled_to_positive_count += 1
                elif idx >= (label_score_num - threshold_idx) and v[0] == -1:
                    self.replaced_label_list.append((k, 0))
                    self.unlabeled_to_negative_count += 1
                else:
                    self.replaced_label_list.append((k, int(v[0])))
        else:
            LOGGER.info("Execute standard mode")
            for idx, (k, v) in enumerate(label_score_list):
                if idx < threshold_idx and v[0] == 0:
                    self.replaced_label_list.append((k, 1))
                    self.converted_unlabeled_count += 1
                else:
                    self.replaced_label_list.append((k, int(v[0])))

    def quantity_processing(self, label_score_table):
        LOGGER.info("Switch quantity strategy")
        label_score_list = list(label_score_table.collect())
        label_score_list.sort(key=lambda x: x[1][1], reverse=True)

        LOGGER.info("Count unlabeled samples")
        label_list = [v[0] for (_, v) in label_score_list]

        if self.mode == consts.TWO_STEP:
            LOGGER.info("Execute two-step mode")
            unlabeled_count = label_list.count(-1)
            if self.threshold_amount > unlabeled_count:
                LOGGER.warning("Param 'threshold_amount' should no larger than unlabeled count")

            reversed_label_score_list = []
            accumulated_count = 0
            for (k, v) in label_score_list:
                if accumulated_count < int(self.threshold_amount / 2) and v[0] == -1:
                    reversed_label_score_list.append((k, [1, v[1]]))
                    self.unlabeled_to_positive_count += 1
                    accumulated_count += 1
                else:
                    reversed_label_score_list.append((k, [int(v[0]), v[1]]))

            reversed_label_score_list.sort(key=lambda x: x[1][1], reverse=False)
            accumulated_count = 0
            for (k, v) in reversed_label_score_list:
                if accumulated_count < int(self.threshold_amount / 2) and v[0] == -1:
                    self.replaced_label_list.append((k, 0))
                    self.unlabeled_to_negative_count += 1
                    accumulated_count += 1
                else:
                    self.replaced_label_list.append((k, int(v[0])))
        else:
            LOGGER.info("Execute standard mode")
            unlabeled_count = label_list.count(0)
            if self.threshold_amount > unlabeled_count:
                LOGGER.warning("Param 'threshold_amount' should no larger than unlabeled count")

            accumulated_count = 0
            for idx, (k, v) in enumerate(label_score_list):
                if accumulated_count < self.threshold_amount and v[0] == 0:
                    self.replaced_label_list.append((k, 1))
                    self.converted_unlabeled_count += 1
                    accumulated_count += 1
                else:
                    self.replaced_label_list.append((k, int(v[0])))

    def probability_processing(self, label_score_table):
        LOGGER.info("Switch probability strategy")
        if self.mode == consts.TWO_STEP:
            LOGGER.info("Execute two-step mode")
            if self.threshold_proba < 0.5:
                LOGGER.warning("Param 'threshold_proba' should no less than 0.5 in two-step mode")

            def replaced_func(x):
                if x[1] >= self.threshold_proba and x[0] == -1:
                    return 1
                elif x[1] <= 1 - self.threshold_proba and x[0] == -1:
                    return 0
                else:
                    return x[0]

            def summarized_func(r, l):
                if r == 1 and l[0] == -1:
                    return 1
                elif r == 0 and l[0] == -1:
                    return 0
                else:
                    return -1

            replaced_label_table = label_score_table.mapValues(replaced_func)
            summary_table = replaced_label_table.join(label_score_table, summarized_func)
            self.unlabeled_to_positive_count = summary_table.filter(lambda k, v: v == 1).count()
            self.unlabeled_to_negative_count = summary_table.filter(lambda k, v: v == 0).count()
        else:
            LOGGER.info("Execute standard mode")

            def replaced_func(x):
                if x[1] >= self.threshold_proba and x[0] == 0:
                    return 1
                else:
                    return x[0]

            def summarized_func(r, l):
                if r == 1 and l[0] == 0:
                    return 1
                else:
                    return 0

            replaced_label_table = label_score_table.mapValues(replaced_func)
            summary_table = replaced_label_table.join(label_score_table, summarized_func)
            self.converted_unlabeled_count = summary_table.filter(lambda k, v: v == 1).count()
        return replaced_label_table

    def interval_processing(self, label_score_table):
        LOGGER.info("Switch interval strategy")
        if self.mode == consts.TWO_STEP:
            LOGGER.info("Execute two-step mode")
            label_score_list = list(label_score_table.collect())
            label_score_list.sort(key=lambda x: x[1][1], reverse=True)

            LOGGER.info("Compute interval boundary")
            idx_list = [idx for idx, (_, v) in enumerate(label_score_list) if v[0] == 1]
            start_idx, end_idx = min(idx_list), max(idx_list)

            for idx, (k, v) in enumerate(label_score_list):
                if idx < start_idx and v[0] == -1:
                    self.replaced_label_list.append((k, 1))
                    self.unlabeled_to_positive_count += 1
                elif idx > end_idx and v[0] == -1:
                    self.replaced_label_list.append((k, 0))
                    self.unlabeled_to_negative_count += 1
                else:
                    self.replaced_label_list.append((k, int(v[0])))
        else:
            raise ValueError("Interval strategy only adapted to two-step mode")

    def replace_labels(self, labeling_strategy, label_score_table):
        if labeling_strategy == consts.PROPORTION:
            self.proportion_processing(label_score_table)
        elif labeling_strategy == consts.QUANTITY:
            self.quantity_processing(label_score_table)
        elif labeling_strategy == consts.PROBABILITY:
            return self.probability_processing(label_score_table)
        else:
            self.interval_processing(label_score_table)

    def callback_info(self):
        if self.mode == consts.TWO_STEP:
            self.add_summary("all", {"count of unlabeled to positive": self.unlabeled_to_positive_count,
                                     "count of unlabeled to negative": self.unlabeled_to_negative_count})
            self.callback_metric(metric_name=self.metric_name,
                                 metric_namespace=self.metric_namespace,
                                 metric_data=[Metric("count of unlabeled to positive",
                                                     self.unlabeled_to_positive_count),
                                              Metric("count of unlabeled to negative",
                                                     self.unlabeled_to_negative_count)])
        else:
            self.add_summary("count of converted unlabeled", self.converted_unlabeled_count)
            self.callback_metric(metric_name=self.metric_name,
                                 metric_namespace=self.metric_namespace,
                                 metric_data=[Metric("count of converted unlabeled",
                                                     self.converted_unlabeled_count)])

    def fit(self, data_insts):
        LOGGER.info("Convert labels by positive unlabeled transformer")
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

            LOGGER.info("Replace labels in view of labeling method")
            if self.labeling_strategy != consts.PROBABILITY:
                self.replace_labels(labeling_strategy=self.labeling_strategy,
                                    label_score_table=label_score_table)
                replaced_label_table = computing_session.parallelize(self.replaced_label_list,
                                                                     include_key=True,
                                                                     partition=intersect_table.partitions)
            else:
                replaced_label_table = self.replace_labels(labeling_strategy=self.labeling_strategy,
                                                           label_score_table=label_score_table)

            LOGGER.info("Construct replaced label feature table")
            replaced_label_feature_table = self.to_data_inst(feature_table, replaced_label_table)
            replaced_label_feature_table = self.assign_properties(replaced_label_feature_table, intersect_table)
            replaced_label_feature_table.schema = intersect_table.schema

            self.callback_info()
            return replaced_label_feature_table

        elif self.role == consts.HOST:
            LOGGER.info("Identify intersect table")
            if data_insts_list[0]:
                intersect_table = data_insts_list[0]
            else:
                intersect_table = data_insts_list[1]
            return intersect_table
