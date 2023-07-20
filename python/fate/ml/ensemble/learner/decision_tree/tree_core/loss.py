import numpy as np
import pandas as pd
from fate.interface import Dataframe


FLOAT_ZERO = 1e-8


def apply_weight(loss: Dataframe, weight: Dataframe):
    return loss['loss'] * weight['weight']


class BCELoss(object):

    @staticmethod
    def initialize(label: Dataframe):
        init_score = label.create_frame()
        init_score['score'] = 0
        return init_score

    @staticmethod
    def predict(score: Dataframe):
        score['score'] = score['score'].sigmoid()
        return score

    @staticmethod
    def compute_loss(label: Dataframe, pred: Dataframe, weight: Dataframe = None):
        
        loss_col = label.create_frame()
        sample_num = len(label)
        label_pred = label.hstack(pred)
        loss_col['loss'] = label_pred.apply_row(lambda s: -(s['label'] * np.log(s['score']) + (1 - s['label']) * np.log(1 - s['score'])))
        loss_col['loss'].fillna(1)
        if weight:
            loss_col['loss'] = apply_weight(loss_col, weight)
        reduce_loss = loss_col['loss'].sum() / sample_num

        return reduce_loss

    @staticmethod
    def compute_grad(gh: Dataframe, label: Dataframe, score: Dataframe):
        gh['g'] = label['label'] - score['score']

    @staticmethod
    def compute_hess(gh: Dataframe, label: Dataframe, score: Dataframe):
        gh['h'] = score * (1 - score)


class CELoss(object):

    @staticmethod
    def initialize(label, class_num=3):
        init_score = label.create_frame()
        init_score[['class_{}'.format(i) for i in range(class_num)]] = [0.0 for i in range(class_num)]
        return init_score

    @staticmethod
    def predict(score: Dataframe):
        pred = score.apply_row(lambda s: np.exp(s)/np.exp(s).sum())
        return pred
 
    @staticmethod
    def compute_loss(label: Dataframe, pred: Dataframe, weight: Dataframe):
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
    def compute_grad(gh: Dataframe, label: Dataframe, score: Dataframe):
        gh['g'] = score - 1

    @staticmethod
    def compute_hess(gh: Dataframe, y, score):
        gh['h'] = score * (1 - score)


class L2Loss(object):

    @staticmethod
    def initialize(label):
        init_score = label.create_frame()
        init_score['score'] = label.mean()
        return init_score

    @staticmethod
    def predict(score):
        return score

    @staticmethod
    def compute_loss(label, pred, weight=None):
        loss_col = label.create_frame()
        sample_num = len(label)
        loss_col['loss'] = (label['label'] - pred['score']) ** 2
        if weight:
            loss_col['loss'] = apply_weight(loss_col, weight)
        reduce_loss = loss_col['loss'].sum() / sample_num
        return reduce_loss

    @staticmethod
    def compute_grad(gh: Dataframe, label, score):
        gh['g'] = 2 * (label['label'] - score['score'])

    @staticmethod
    def compute_hess(gh: Dataframe, label, score):
        gh['h'] = 2


class L1Loss(object):

    @staticmethod
    def initialize(label):
        init_score = label.create_frame()
        init_score['score'] = label.median()
        return init_score

    @staticmethod
    def predict(score):
        return score

    @staticmethod
    def compute_loss(label, pred, weight=None):
        loss_col = label.create_frame()
        sample_num = len(label)
        loss_col['loss'] = (label['label'] - pred['score']).abs()
        if weight:
            loss_col['loss'] = apply_weight(loss_col, weight)
        reduce_loss = loss_col['loss'].sum() / sample_num
        return reduce_loss

    @staticmethod
    def compute_grad(gh: Dataframe, label, score):
        gh['g'] = label['label'] - score['score']
        def compute_l1_g(s):
            ret = 0.0
            if s[0] > FLOAT_ZERO:
                ret = 1.0
            elif s[0] < FLOAT_ZERO:
                ret = -1.0
            ret = pd.Series({'g': ret})
            return ret
        gh['g'] = gh['g'].apply_row(compute_l1_g)

    @staticmethod
    def compute_hess(gh: Dataframe, label, score):
        gh['h'] = 1