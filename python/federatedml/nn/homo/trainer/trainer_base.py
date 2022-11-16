import abc
import importlib
import json
import tempfile
from typing import List

import numpy as np
import torch as t
import torch.optim
from torch.nn import Module

from federatedml.evaluation.evaluation import Evaluation
from federatedml.feature.instance import Instance
from federatedml.model_base import Metric, MetricMeta
from federatedml.model_base import serialize_models
from federatedml.nn.backend.utils.common import ML_PATH, get_homo_model_dict
from federatedml.param import EvaluateParam
from federatedml.protobuf.generated.homo_nn_model_meta_pb2 import HomoNNMeta
from federatedml.protobuf.generated.homo_nn_model_param_pb2 import HomoNNParam
from federatedml.util import LOGGER
from federatedml.util import consts


class StdReturnFormat(object):

    def __init__(self, id_table_list, pred_table, classes):
        self.id = id_table_list
        self.pred_table = pred_table
        self.classes = classes

    def __call__(self, ):
        return self.id, self.pred_table, self.classes


class TrainerBase(object):

    def __init__(self, **kwargs):

        self._fed_mode = True
        self.role = None
        self.party_id = None
        self.party_id_list = None
        self._flowid = None
        self._cache_model = None
        self._model = None
        self._tracker = None
        self._model_checkpoint = None
        self._check_point_history = []
        self._summary = {}

        # nn config
        self.nn_define, self.opt_define, self.loss_define = {}, {}, {}

    @staticmethod
    def is_pos_int(val):
        return val > 0 and isinstance(val, int)

    @staticmethod
    def is_float(val):
        return isinstance(val, float)

    @staticmethod
    def is_bool(val):
        return isinstance(val, bool)

    @staticmethod
    def check_trainer_param(
            var_list,
            name_list,
            judge_func,
            warning_str,
            allow_none=True):
        for var, name in zip(var_list, name_list):
            if allow_none and var is None:
                continue
            assert judge_func(var), warning_str.format(name)

    @property
    def model(self):
        if not hasattr(self, '_model'):
            raise AttributeError(
                'model variable is not initialized, remember to call'
                ' super(your_class, self).__init__()')
        if self._model is None:
            raise AttributeError(
                'model is not set, use set_model() function to set training model')

        return self._model

    @model.setter
    def model(self, val):
        self._model = val

    @property
    def fed_mode(self):
        if not hasattr(self, '_fed_mode'):
            raise AttributeError(
                'run_local_mode variable is not initialized, remember to call'
                ' super(your_class, self).__init__()')
        return self._fed_mode

    @fed_mode.setter
    def fed_mode(self, val):
        self._fed_mode = val

    def local_mode(self):
        self.fed_mode = False

    def set_nn_config(self, nn_define, optimizer_define, loss_define):
        self.nn_define = nn_define
        self.opt_define = optimizer_define
        self.loss_define = loss_define

    def set_tracker(self, tracker):
        self._tracker = tracker

    def init_checkpoint(self, chkp):
        self._model_checkpoint = chkp

    def set_flowid(self, flowid):
        """
        Set flow id, and initialize transfer variable
        """
        self._flowid = flowid

    def set_role(self, role):
        """
        set self role
        """
        self.role = role

    def set_party_id(self, party_id):
        self.party_id = party_id

    def set_party_id_list(self, party_id_list):
        self.party_id_list = party_id_list

    def set_model(self, model: Module):
        if not issubclass(type(model), Module):
            raise ValueError('model must be a subclass of pytorch nn.Module')
        self.model = model

    def get_checkpoint_history(self):
        return self._check_point_history

    def _get_model_param_and_meta(self, model, optimizer=None, epoch_idx=-1):

        if issubclass(type(model), Module):
            self._cache_model = model
            opt_state_dict = None
            if optimizer is not None:
                assert isinstance(optimizer, torch.optim.Optimizer), \
                    'optimizer must be an instance of torch.optim.Optimizer'
                opt_state_dict = optimizer.state_dict()

            model_status = {
                'model': model.state_dict(),
                'optimizer': opt_state_dict,
            }

            with tempfile.TemporaryFile() as f:
                torch.save(model_status, f)
                f.seek(0)
                model_saved_bytes = f.read()

            param = HomoNNParam()
            meta = HomoNNMeta()

            param.model_bytes = model_saved_bytes
            meta.nn_define.append(json.dumps(self.nn_define))
            meta.optimizer_define.append(json.dumps(self.opt_define))
            meta.loss_func_define.append(json.dumps(self.loss_define))

            return param, meta

        else:
            raise ValueError(
                'export model must be a subclass of torch nn.Module, however got {}'.format(
                    type(model)))

    def export_model(self, model, optimizer=None, epoch_idx=-1):

        param, meta = self._get_model_param_and_meta(
            model, optimizer, epoch_idx)
        self._cache_model = (param, meta)

    @staticmethod
    def task_type_infer(predict_result: t.Tensor, true_label):

        # infer task type and classes(of classification task)
        predict_result = predict_result.cpu()
        true_label = true_label.cpu()
        pred_shape = predict_result.shape

        with t.no_grad():

            if true_label.max() == 1.0 and true_label.min() == 0.0:
                return consts.BINARY

            if (len(pred_shape) > 1) and (pred_shape[1] > 1):
                if t.isclose(
                    predict_result.sum(
                        axis=1).cpu(), t.Tensor(
                        [1.0])).all():
                    return consts.MULTY
                else:
                    return None
            elif (len(pred_shape) == 1) or (pred_shape[1] == 1):
                # if t.max(predict_result) <= 1.0 and t.min(predict_result) >= 0.0:
                #     return consts.BINARY
                # else:
                return consts.REGRESSION

        return None

    def format_predict_result(self, sample_ids: List, predict_result: t.Tensor,
                              true_label: t.Tensor, task_type: str = None):

        predict_result = predict_result.cpu().detach()

        if task_type == 'auto':
            task_type = self.task_type_infer(predict_result, true_label)
            if task_type is None:
                LOGGER.warning(
                    'unable to infer predict result type, predict process will be skipped')
                return None

        classes = None
        if task_type == consts.BINARY:
            classes = [0, 1]
        elif task_type == consts.MULTY:
            classes = [i for i in range(predict_result.shape[1])]

        true_label = true_label.cpu().detach().flatten().tolist()

        if task_type == consts.MULTY:
            predict_result = predict_result.tolist()
        else:
            predict_result = predict_result.flatten().tolist()

        id_table = [(id_, Instance(label=l))
                    for id_, l in zip(sample_ids, true_label)]
        score_table = [(id_, pred)
                       for id_, pred in zip(sample_ids, predict_result)]
        return StdReturnFormat(id_table, score_table, classes)

    def get_cached_model(self):
        return self._cache_model

    def set_checkpoint(self, model, optimizer=None, epoch_idx=-1):

        assert isinstance(
            epoch_idx, int) and epoch_idx >= 0, 'epoch idx must be an int >= 0'
        if self._model_checkpoint:
            param, meta = self._get_model_param_and_meta(
                model, optimizer, epoch_idx)
            model_dict = get_homo_model_dict(param, meta)
            self._model_checkpoint.add_checkpoint(
                epoch_idx, to_save_model=serialize_models(model_dict))
            if not self._check_point_history:
                self._check_point_history = []
            self._check_point_history.append(epoch_idx)
            LOGGER.info('check point at epoch {} saved'.format(epoch_idx))

    def callback_metric(self, metric_name: str, value: float, metric_type='train', epoch_idx=0):

        assert metric_type in [
            'train', 'validate'], 'metric_type should be train or validate'
        iter_name = 'iteration_{}'.format(epoch_idx)
        if self._tracker is not None:
            self._tracker.log_metric_data(
                metric_type, iter_name, [
                    Metric(
                        metric_name, np.round(
                            value, 6))])
            self._tracker.set_metric_meta(
                metric_type, iter_name, MetricMeta(
                    name=metric_name, metric_type='EVALUATION_SUMMARY'))

    def callback_loss(self, loss: float, epoch_idx: int):

        if self._tracker is not None:
            self._tracker.log_metric_data(
                metric_name="loss",
                metric_namespace="train",
                metrics=[Metric(epoch_idx, loss)],
            )

    def evaluation(self, sample_ids: list, pred_scores: t.Tensor, label: t.Tensor, dataset_type='train',
                   metric_list=None, epoch_idx=0, task_type=None):

        eval_obj = Evaluation()
        if task_type == 'auto':
            task_type = self.task_type_infer(pred_scores, label)

        if task_type is None:
            return

        assert dataset_type in [
            'train', 'validate'], 'dataset_type must in ["train", "validate"]'

        eval_param = EvaluateParam(eval_type=task_type)
        if task_type == consts.BINARY:
            eval_param.metrics = ['auc', 'ks']
        elif task_type == consts.MULTY:
            eval_param.metrics = ['accuracy', 'precision', 'recall']

        eval_param.check_single_value_default_metric()
        eval_obj._init_model(eval_param)

        pred_scores = pred_scores.cpu().detach().numpy()
        label = label.cpu().detach().numpy()

        if task_type == consts.REGRESSION or task_type == consts.BINARY:
            pred_scores = pred_scores.flatten()
            label = label.flatten()

        pred_scores = pred_scores.tolist()
        label = label.tolist()

        eval_data = []
        for id_, s, l in zip(sample_ids, pred_scores, label):
            if task_type == consts.REGRESSION:
                eval_data.append([id_, (l, s, s)])
            if task_type == consts.MULTY:
                pred_label = np.argmax(s)
                eval_data.append([id_, (l, pred_label, s)])
            elif task_type == consts.BINARY:
                pred_label = (s > 0.5) + 1
                eval_data.append([id_, (l, pred_label, s)])

        eval_result = eval_obj.evaluate_metrics(dataset_type, eval_data)

        if self._tracker is not None:
            eval_obj.set_tracker(self._tracker)
            # send result to fate-board
            eval_obj.callback_metric_data(
                {'iteration_{}'.format(epoch_idx): [eval_result]})

    @abc.abstractmethod
    def train(self, train_set, validate_set=None, optimizer=None, loss=None):
        """
            train_set : A Dataset Instance, must be a instance of subclass of Dataset (federatedml.nn.dataset.base),
                      for example, TableDataset() (from federatedml.nn.dataset.table)

            validate_set : A Dataset Instance, but optional must be a instance of subclass of Dataset
                    (federatedml.nn.dataset.base), for example, TableDataset() (from federatedml.nn.dataset.table)

            optimizer : A pytorch optimizer class instance, for example, t.optim.Adam(), t.optim.SGD()

            loss : A pytorch Loss class, for example, nn.BECLoss(), nn.CrossEntropyLoss()
        """
        pass

    @abc.abstractmethod
    def predict(self, dataset):
        pass


def get_trainer_class(trainer_module_name: str):
    if trainer_module_name.endswith('.py'):
        trainer_module_name = trainer_module_name.replace('.py', '')
    ds_modules = importlib.import_module(
        '{}.homo.trainer.{}'.format(
            ML_PATH, trainer_module_name))
    try:

        for k, v in ds_modules.__dict__.items():
            if isinstance(v, type):
                if issubclass(v, TrainerBase) and v is not TrainerBase:
                    return v
        raise ValueError('Did not find any class in {}.py that is the subclass of Trainer class'.
                         format(trainer_module_name))
    except ValueError as e:
        raise e
