import numpy as np
import pandas as pd
import torch as t
from fate.arch.dataframe import DataFrame
from scipy.special import expit as sigmoid





def apply_weight(loss: DataFrame, weight: DataFrame):
    return loss['loss'] * weight['weight']


class BCELoss(object):

    @staticmethod
    def initialize(label: DataFrame):
        init_score = label.create_frame()
        init_score['score'] = 0.0
        return init_score

    @staticmethod
    def predict(score: DataFrame):
        pred_rs = score.create_frame()
        pred_rs['score'] = score.apply_row(lambda s: sigmoid(s))
        return pred_rs

    @staticmethod
    def compute_loss(label: DataFrame, pred: DataFrame):
        
        sample_num = len(label)
        label_pred = DataFrame.hstack([label, pred])
        label_pred['loss'] = label_pred.apply_row(lambda s: -(s[0] * np.log(s[1]) + (1 - s[0]) * np.log(1 - s[1])), with_label=True)
        loss_rs = label_pred['loss'].fillna(1)
        reduce_loss = loss_rs['loss'].sum() / sample_num
        return reduce_loss

    @staticmethod
    def compute_grad(gh: DataFrame, label: DataFrame, predict_score: DataFrame):
        gh['g'] = predict_score - label

    @staticmethod
    def compute_hess(gh: DataFrame, label: DataFrame, predict_score: DataFrame):
        gh['h'] = predict_score * (1 - predict_score)


class CELoss(object):

    @staticmethod
    def initialize(label, class_num=3):
        init_score = label.create_frame()
        init_score['score'] = [0 for i in range(class_num)]
        return init_score

    @staticmethod
    def predict(score: DataFrame):
        pred = score.apply_row(lambda s: np.exp(s)/np.exp(s).sum())
        return pred
 
    @staticmethod
    def compute_loss(label: DataFrame, pred: DataFrame, weight: DataFrame):
        loss_col = label.create_frame()
        label_pred = label.hstack(pred)
        sample_num = len(label)
        loss_col['loss'] = label_pred.apply_row(lambda s: np.log(s[1:][int(s[0])]))
        loss_col['loss'].fillna(1)
        if weight:
            loss_col['loss'] = apply_weight(loss_col, weight)
        reduce_loss = loss_col['loss'].sum() / sample_num
        return reduce_loss

    @staticmethod
    def compute_grad(gh: DataFrame, label: DataFrame, score: DataFrame):
        gh['g'] = score - 1

    @staticmethod
    def compute_hess(gh: DataFrame, y, score):
        gh['h'] = score * (1 - score)


class L2Loss(object):

    @staticmethod
    def initialize(label):
        init_score = label.create_frame()
        mean_score = float(label.mean())
        init_score['score'] = mean_score
        return init_score, mean_score

    @staticmethod
    def predict(score):
        return score

    @staticmethod
    def compute_loss(label: DataFrame, pred: DataFrame):
        loss_col = label.create_frame()
        sample_num = len(label)
        loss_col['loss'] = (label - pred['score']) ** 2
        reduce_loss = loss_col['loss'].sum() / sample_num
        return reduce_loss

    @staticmethod
    def compute_grad(gh: DataFrame, label, score):
        gh['g'] = 2 * (label - score['score'])

    @staticmethod
    def compute_hess(gh: DataFrame, label, score):
        gh['h'] = 2


# class L1Loss(object):

#     @staticmethod
#     def initialize(label):
#         init_score = label.create_frame()
#         init_score['score'] = label.median()
#         return init_score

#     @staticmethod
#     def predict(score):
#         return score

#     @staticmethod
#     def compute_loss(label, pred, weight=None):
#         loss_col = label.create_frame()
#         sample_num = len(label)
#         loss_col['loss'] = (label['label'] - pred['score']).abs()
#         if weight:
#             loss_col['loss'] = apply_weight(loss_col, weight)
#         reduce_loss = loss_col['loss'].sum() / sample_num
#         return reduce_loss

#     @staticmethod
#     def compute_grad(gh: DataFrame, label, score):
#         gh['g'] = label['label'] - score['score']
#         def compute_l1_g(s):
#             ret = 0.0
#             if s[0] > FLOAT_ZERO:
#                 ret = 1.0
#             elif s[0] < FLOAT_ZERO:
#                 ret = -1.0
#             ret = pd.Series({'g': ret})
#             return ret
#         gh['g'] = gh['g'].apply_row(compute_l1_g)

#     @staticmethod
#     def compute_hess(gh: DataFrame, label, score):
#         gh['h'] = 1