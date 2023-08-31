import os
import abc
import importlib

import torch as t
import numpy as np
from torch.nn import Module
from typing import List
from federatedml.util import consts
from federatedml.util import LOGGER
from federatedml.model_base import serialize_models
from federatedml.nn.backend.utils.common import ML_PATH, LLM_PATH
from federatedml.feature.instance import Instance
from federatedml.evaluation.evaluation import Evaluation
from federatedml.model_base import Metric, MetricMeta
from federatedml.param import EvaluateParam


class StdReturnFormat(object):

    def __init__(self, id_table_list, pred_table, classes):
        self.id = id_table_list
        self.pred_table = pred_table
        self.classes = classes

    def __call__(self,):
        return self.id, self.pred_table, self.classes


class ExporterBase(object):

    def __init__(self, *args, **kwargs):
        pass

    def export_model_dict(self, model=None, optimizer=None, model_define=None, optimizer_define=None, loss_define=None,
                          epoch_idx=-1, converge_status=False, loss_history=None, best_epoch=-1, extra_data={}):
        pass


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
        self._exporter = None
        self._evaluation_summary = {}
        self._client_num = None
        self._optimizer = None
        self._loss_fn = None


        # running status
        self._set_model_checkpoint_epoch = set()

        # nn config
        self.nn_define, self.opt_define, self.loss_define = {}, {}, {}

        # ret summary
        self._summary = {}

        # deepspeed enabled
        self._enable_deepspeed = False
        self._deepspeed_zero_3 = False

        # deepspeed config
        self._ds_config = None

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
        assert isinstance(val, bool), 'fed mode must be a bool'
        self._fed_mode = val

    def enable_deepspeed(self, ds_config, is_zero_3=False):
        self._ds_config = ds_config
        self._enable_deepspeed = True
        self._deepspeed_zero_3 = is_zero_3

    def local_mode(self):
        self.fed_mode = False

    def set_nn_config(self, nn_define, optimizer_define, loss_define):
        self.nn_define = nn_define
        self.opt_define = optimizer_define
        self.loss_define = loss_define

    def set_tracker(self, tracker):
        self._tracker = tracker

    def set_checkpoint(self, chkp):
        self._model_checkpoint = chkp

    def set_party_id_list(self, party_id_list):
        self.party_id_list = party_id_list

    def set_model_exporter(self, exporter):
        assert isinstance(
            exporter, ExporterBase), 'exporter is not an instance of ExporterBase'
        self._exporter = exporter

    def get_cached_model(self):
        return self._cache_model

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
                return consts.REGRESSION

        return None

    def _update_metric_summary(self, metric_dict):

        if len(metric_dict) == 0:
            return

        iter_name = list(metric_dict.keys())[0]
        metric_dict = metric_dict[iter_name]

        if len(self._evaluation_summary) == 0:
            self._evaluation_summary = {namespace: {}
                                        for namespace in metric_dict}

        for namespace in metric_dict:
            for metric_name in metric_dict[namespace]:
                epoch_metric = metric_dict[namespace][metric_name]
                if namespace not in self._evaluation_summary:
                    self._evaluation_summary[namespace] = {}
                if metric_name not in self._evaluation_summary[namespace]:
                    self._evaluation_summary[namespace][metric_name] = []
                self._evaluation_summary[namespace][metric_name].append(
                    epoch_metric)

    def get_evaluation_summary(self):
        return self._evaluation_summary

    def get_summary(self):
        return self._summary

    """
    User Interfaces
    """

    def _local_save(
            self,
            model,
            optimizer,
            epoch_idx,
            converge_status,
            loss_history,
            best_epoch,
            extra_data,
            save_path):

        LOGGER.debug('save model to local dir')
        if hasattr(model, "enable_save_pretrained") and model.enable_save_pretrained:
            model.save_pretrained(save_path)
        else:
            unwrap_model = TrainerBase.unwrap_model(model)
            if hasattr(model, "enable_save_pretrained") and model.enable_save_pretrained:
                unwrap_model.save_pretrained(save_path)
            else:
                if model is None:
                    model_state_dict = None
                else:
                    model_state_dict = model.state_dict()
                if optimizer is None:
                    optimizer_state_dict = None
                else:
                    optimizer_state_dict = optimizer.state_dict()
                model_dict = {
                    'model': model_state_dict,
                    'optimizer': optimizer_state_dict,
                    'model_define': self.nn_define,
                    'optimizer_define': self.opt_define,
                    'loss_define': self.loss_define,
                    'epoch_idx': epoch_idx,
                    'converge_status': converge_status,
                    'loss_history': loss_history,
                    'best_epoch': best_epoch,
                    'extra_data': extra_data
                }
                LOGGER.info('save path is {}'.format(save_path))
                t.save(model_dict, save_path)

        local_save_path = save_path if not self._enable_deepspeed else os.environ[consts.FLOW_MODEL_SYNC_PATH]
        model_dict = self._exporter.export_model_dict(model_define=self.nn_define,
                                                      optimizer_define=self.opt_define,
                                                      loss_define=self.loss_define,
                                                      epoch_idx=epoch_idx,
                                                      converge_status=converge_status,
                                                      loss_history=loss_history,
                                                      best_epoch=best_epoch,
                                                      extra_data=extra_data,
                                                      local_save_path=local_save_path
                                                      )
        self._cache_model = model_dict

    def set_model(self, model: Module):
        if not issubclass(type(model), Module):
            raise ValueError('model must be a subclass of pytorch nn.Module')
        self.model = model

    def save(
            self,
            model=None,
            epoch_idx=-1,
            optimizer=None,
            converge_status=False,
            loss_history=None,
            best_epoch=-1,
            extra_data={}):

        assert isinstance(
            epoch_idx, int) and epoch_idx >= 0, 'epoch idx must be an int >= 0'

        if self._exporter:
            LOGGER.debug('save model to fate')
            model_dict = self._exporter.export_model_dict(model=model,
                                                          optimizer=optimizer,
                                                          model_define=self.nn_define,
                                                          optimizer_define=self.opt_define,
                                                          loss_define=self.loss_define,
                                                          epoch_idx=epoch_idx,
                                                          converge_status=converge_status,
                                                          loss_history=loss_history,
                                                          best_epoch=best_epoch,
                                                          extra_data=extra_data
                                                          )
            self._cache_model = model_dict

    def checkpoint(
            self,
            model=None,
            epoch_idx=-1,
            optimizer=None,
            converge_status=False,
            loss_history=None,
            best_epoch=-1,
            extra_data={}):

        assert isinstance(
            epoch_idx, int) and epoch_idx >= 0, 'epoch idx must be an int >= 0'

        """
        if isinstance(TrainerBase.unwrap_model(model), PELLM):
            raise ValueError("save checkpoint of Pretrained model should provide local dir")
        """

        if self._model_checkpoint:

            if self._exporter is None:
                raise RuntimeError('exporter is None, cannot save checkpoint')

            if epoch_idx in self._set_model_checkpoint_epoch:
                LOGGER.info(
                    'checkpoint at epoch {} set, skip setting checkpoint'.format(epoch_idx))
                return

            self.save(model=model, epoch_idx=epoch_idx, optimizer=optimizer, converge_status=converge_status,
                      loss_history=loss_history, best_epoch=best_epoch, extra_data=extra_data)

            self._model_checkpoint.add_checkpoint(len(self._set_model_checkpoint_epoch),
                                                  to_save_model=serialize_models(self._cache_model))  # step_index, to_save_model
            self._set_model_checkpoint_epoch.add(epoch_idx)
            LOGGER.info('checkpoint at epoch {} saved'.format(epoch_idx))

    def local_save(self,
                   model=None,
                   epoch_idx=-1,
                   optimizer=None,
                   converge_status=False,
                   loss_history=None,
                   best_epoch=-1,
                   extra_data={}):

        assert isinstance(
            epoch_idx, int) and epoch_idx >= 0, 'epoch idx must be an int >= 0'

        if self._exporter:
            # default saving folder is under the job folder
            model_name = "model.pkl"
            if self._enable_deepspeed:
                save_path = os.path.join(os.environ[consts.DEEPSPEED_MODEL_DIR], model_name)
            else:
                save_path = os.path.abspath(os.path.join('../../../../', model_name))

            self._local_save(
                model,
                optimizer,
                epoch_idx,
                converge_status,
                loss_history,
                best_epoch,
                extra_data,
                save_path)

    def local_checkpoint(self,
                         model=None,
                         epoch_idx=-1,
                         optimizer=None,
                         converge_status=False,
                         loss_history=None,
                         best_epoch=-1,
                         extra_data={}):

        if self._exporter:
            # default saving folder is under the job folder
            model_name = 'checkpoint_{}.pkl'.format(epoch_idx)
            if self._enable_deepspeed:
                save_path = os.path.join(os.environ[consts.DEEPSPEED_MODEL_DIR], model_name)
            else:
                save_path = os.path.abspath(os.path.join('../../../../', model_name))
            self._local_save(
                model,
                optimizer,
                epoch_idx,
                converge_status,
                loss_history,
                best_epoch,
                extra_data,
                save_path)
            self._model_checkpoint.add_checkpoint(len(self._set_model_checkpoint_epoch),
                                                  to_save_model=serialize_models(self._cache_model))  # step_index, to_save_model
            self._set_model_checkpoint_epoch.add(epoch_idx)
            LOGGER.info('checkpoint at epoch {} saved'.format(epoch_idx))

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

    def summary(self, summary_dict: dict):

        assert isinstance(summary_dict, dict), 'summary must be a dict'
        self._summary = summary_dict

    def evaluation(self, sample_ids: list, pred_scores: t.Tensor, label: t.Tensor, dataset_type='train',
                   metric_list=None, epoch_idx=0, task_type=None):

        eval_obj = Evaluation()
        if task_type == 'auto':
            task_type = self.task_type_infer(pred_scores, label)

        if task_type is None:
            LOGGER.debug('cannot infer task type, return')
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
        label = label.cpu().detach().numpy().flatten()

        if task_type == consts.REGRESSION or task_type == consts.BINARY:
            pred_scores = pred_scores.flatten()
            label = label.flatten()

        pred_scores = pred_scores.tolist()
        label = label.tolist()
        assert len(pred_scores) == len(
            label), 'the length of predict score != the length of label, pred {} and label {}'.format(len(pred_scores), len(label))
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

        self._update_metric_summary(eval_obj.metric_summaries)
        return self._evaluation_summary

    def to_cuda(self, var, device=0):
        if hasattr(var, 'cuda'):
            return var.cuda(device)
        elif isinstance(var, tuple) or isinstance(var, list):
            ret = tuple(self.to_cuda(i) for i in var)
            return ret
        elif isinstance(var, dict):
            for k in var:
                if hasattr(var[k], 'cuda'):
                    var[k] = var[k].cuda(device)
            return var
        else:
            return var

    @abc.abstractmethod
    def train(self, train_set, validate_set=None, optimizer=None, loss=None, extra_data={}):
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

    @abc.abstractmethod
    def server_aggregate_procedure(self, extra_data={}):
        pass

    @staticmethod
    def unwrap_model(model):
        if hasattr(model, "module"):
            return TrainerBase.unwrap_model(model.module)
        else:
            return model


"""
Load Trainer
"""


def get_trainer_class(trainer_module_name: str):
    if trainer_module_name.endswith('.py'):
        trainer_module_name = trainer_module_name.replace('.py', '')

    std_fate_trainer_path = '{}.homo.trainer.{}'.format(ML_PATH, trainer_module_name)

    paths_to_check = [std_fate_trainer_path]
    errors = []
    try:
        importlib.import_module(LLM_PATH)
        fate_llm_trainer_path = '{}.trainer.{}'.format(LLM_PATH, trainer_module_name)
        paths_to_check.append(fate_llm_trainer_path)
    except Exception as e:
        pass

    trainers = []
    ds_modules = None

    for path in paths_to_check:
        try:
            ds_modules = importlib.import_module(path)
            break
        except Exception as e:
            errors.append(str(e))

    if ds_modules is None:
        raise ImportError(
            'Could not import from any of the paths: {}, error details {}'.format(
                ', '.join(paths_to_check), errors))

    for k, v in ds_modules.__dict__.items():

        if isinstance(v, type):
            if issubclass(v, TrainerBase) and v is not TrainerBase:
                trainers.append(v)

    if len(trainers) == 0:
        raise ValueError('Did not find any class in {}.py that is the subclass of Trainer class'.
                         format(trainer_module_name))
    else:
        return trainers[-1]  # return the last defined trainer
