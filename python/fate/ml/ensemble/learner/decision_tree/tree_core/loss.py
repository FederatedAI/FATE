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
import pandas as pd
import torch as t
from fate.arch.dataframe import DataFrame
from scipy.special import expit as sigmoid
from fate.ml.utils.predict_tools import BINARY, MULTI, REGRESSION


BINARY_BCE = "binary:bce"
MULTI_CE = "multi:ce"
REGRESSION_L2 = "regression:l2"


def apply_weight(loss: DataFrame, weight: DataFrame):
    return loss["loss"] * weight["weight"]


class BCELoss(object):
    @staticmethod
    def initialize(label: DataFrame):
        init_score = label.create_frame()
        init_score["score"] = 0.0
        return init_score

    @staticmethod
    def predict(score: DataFrame):
        pred_rs = score.create_frame()
        pred_rs["score"] = score.apply_row(lambda s: sigmoid(s))
        return pred_rs

    @staticmethod
    def compute_loss(label: DataFrame, pred: DataFrame):
        sample_num = len(label)
        label_pred = DataFrame.hstack([label, pred])
        label_pred["loss"] = label_pred.apply_row(
            lambda s: -(s[0] * np.log(s[1]) + (1 - s[0]) * np.log(1 - s[1])), with_label=True
        )
        loss_rs = label_pred["loss"].fillna(1)
        reduce_loss = loss_rs["loss"].sum() / sample_num
        return reduce_loss

    @staticmethod
    def compute_grad(gh: DataFrame, label: DataFrame, predict_score: DataFrame):
        gh["g"] = predict_score - label

    @staticmethod
    def compute_hess(gh: DataFrame, label: DataFrame, predict_score: DataFrame):
        gh["h"] = predict_score * (1 - predict_score)


class CELoss(object):
    @staticmethod
    def initialize(label, class_num=3):
        init_score = label.create_frame()
        init_score["score"] = [0.0 for i in range(class_num)]
        return init_score

    @staticmethod
    def predict(score: DataFrame):
        def softmax(s):
            s = np.array(s["score"]).astype(np.float64)
            ret = (np.exp(s) / np.exp(s).sum()).tolist()
            return [ret]

        pred_rs = score.create_frame()
        pred_rs["score"] = score.apply_row(lambda s: softmax(s))
        return pred_rs

    @staticmethod
    def compute_loss(label: DataFrame, pred: DataFrame, weight: DataFrame):
        loss_col = label.create_frame()
        label_pred = label.hstack(pred)
        sample_num = len(label)
        loss_col["loss"] = label_pred.apply_row(lambda s: np.log(s[1:][int(s[0])]))
        loss_col["loss"].fillna(1)
        if weight:
            loss_col["loss"] = apply_weight(loss_col, weight)
        reduce_loss = loss_col["loss"].sum() / sample_num
        return reduce_loss

    @staticmethod
    def compute_grad(gh: DataFrame, label: DataFrame, score: DataFrame):
        gh["g"] = score.apply_row(lambda s: [[i - 1 for i in s["score"]]])

    @staticmethod
    def compute_hess(gh: DataFrame, y, score):
        gh["h"] = score.apply_row(lambda s: [[2 * i * (1 - i) for i in s["score"]]])


class L2Loss(object):
    @staticmethod
    def initialize(label):
        init_score = label.create_frame()
        mean_score = float(label.mean())
        init_score["score"] = mean_score
        return init_score, mean_score

    @staticmethod
    def predict(score):
        return score

    @staticmethod
    def compute_loss(label: DataFrame, pred: DataFrame):
        loss_col = label.create_frame()
        sample_num = len(label)
        loss_col["loss"] = (label - pred["score"]) ** 2
        reduce_loss = loss_col["loss"].sum() / sample_num
        return reduce_loss

    @staticmethod
    def compute_grad(gh: DataFrame, label, score):
        gh["g"] = 2 * (score["score"] - label)

    @staticmethod
    def compute_hess(gh: DataFrame, label, score):
        gh["h"] = 2


OBJECTIVE = {BINARY_BCE: BCELoss, MULTI_CE: CELoss, REGRESSION_L2: L2Loss}


def get_task_info(objective):
    task_type = objective.split(":")[0]
    return task_type
