import os
import time

import torch
import torch as t
import torch.distributed as dist
import tqdm
import numpy as np
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from federatedml.framework.homo.aggregator.secure_aggregator import SecureAggregatorClient as SecureAggClient
from federatedml.framework.homo.aggregator.secure_aggregator import SecureAggregatorServer as SecureAggServer
from federatedml.nn.dataset.base import Dataset
from federatedml.nn.homo.trainer.trainer_base import TrainerBase
from federatedml.util import LOGGER, consts
from federatedml.optim.convergence import converge_func_factory


class FedAVGTrainer(TrainerBase):
    """

    Parameters
    ----------
    epochs: int >0, epochs to train
    batch_size: int, -1 means full batch
    secure_aggregate: bool, default is True, whether to use secure aggregation. if enabled, will add random number
                            mask to local models. These random number masks will eventually cancel out to get 0.
    weighted_aggregation: bool, whether add weight to each local model when doing aggregation.
                         if True, According to origin paper, weight of a client is: n_local / n_global, where n_local
                         is the sample number locally and n_global is the sample number of all clients.
                         if False, simply averaging these models.

    early_stop: None, 'diff' or 'abs'. if None, disable early stop; if 'diff', use the loss difference between
                two epochs as early stop condition, if differences < tol, stop training ; if 'abs', if loss < tol,
                stop training
    tol: float, tol value for early stop

    aggregate_every_n_epoch: None or int. if None, aggregate model on the end of every epoch, if int, aggregate
                             every n epochs.

    cuda: None, int or list of int. if None, use cpu; if int, use the the {int} device, if list of int, use the
          This trainier will automatically detect use DataParallel for multi GPU training, the first index will be
          the main device and the output device.

    pin_memory: bool, for pytorch DataLoader
    shuffle: bool, for pytorch DataLoader
    data_loader_worker: int, for pytorch DataLoader, number of workers when loading data
    validation_freqs: None or int. if int, validate your model and send validate results to fate-board every n epoch.
                      if is binary classification task, will use metrics 'auc', 'ks', 'gain', 'lift', 'precision'
                      if is multi classification task, will use metrics 'precision', 'recall', 'accuracy'
                      if is regression task, will use metrics 'mse', 'mae', 'rmse', 'explained_variance', 'r2_score'
    checkpoint_save_freqs: save model every n epoch, if None, will not save checkpoint.
    task_type: str, 'auto', 'binary', 'multi', 'regression'
               this option decides the return format of this trainer, and the evaluation type when running validation.
               if auto, will automatically infer your task type from labels and predict results.
    save_to_local_dir: bool, if True, a dictionary containing the model, optimizer, and metadata will be saved to a local directory.
                The path is structured as follows: fateflow/jobs/${jobid}/${party}/${party_id}/${your_nn_component}.
                If set to False, the model will not be saved to the FATE framework in protobuf format.
    """

    def __init__(self, epochs=10, batch_size=512,  # training parameter
                 early_stop=None, tol=0.0001,  # early stop parameters
                 secure_aggregate=True, weighted_aggregation=True, aggregate_every_n_epoch=None,  # federation
                 cuda=None,
                 pin_memory=True, shuffle=True, data_loader_worker=0,  # GPU & dataloader
                 validation_freqs=None,  # validation configuration
                 checkpoint_save_freqs=None,  # checkpoint configuration
                 task_type='auto',  # task type
                 save_to_local_dir=False  # save model to local path
                 ):

        super(FedAVGTrainer, self).__init__()

        # training parameters
        self.epochs = epochs
        self.tol = tol
        self.validation_freq = validation_freqs
        self.save_freq = checkpoint_save_freqs
        self.save_to_local_dir = save_to_local_dir

        self.task_type = task_type
        task_type_allow = [
            consts.BINARY,
            consts.REGRESSION,
            consts.MULTY,
            'auto']
        assert self.task_type in task_type_allow, 'task type must in {}'.format(
            task_type_allow)

        # aggregation param
        self.secure_aggregate = secure_aggregate
        self.weighted_aggregation = weighted_aggregation
        self.aggregate_every_n_epoch = aggregate_every_n_epoch

        # GPU, check cuda setting
        self.cuda = cuda
        self.cuda_main_device = None
        self.data_parallel = False
        self.parallel_model = None

        if not torch.cuda.is_available() and self.cuda is not None:
            raise ValueError('Cuda is not available on this machine')
        if isinstance(self.cuda, int):
            self.cuda_main_device = self.cuda
        elif isinstance(self.cuda, list):
            for i in self.cuda:
                assert isinstance(i, int), 'cuda device must be int, but got {}'.format(self.cuda)
            self.cuda_main_device = self.cuda[0]
            if len(self.cuda) > 1:
                self.data_parallel = True
                LOGGER.info('Using DataParallel in Pytorch')

        # data loader
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.data_loader_worker = data_loader_worker
        self.data_loader = None

        self.early_stop = early_stop
        early_stop_type = ['diff', 'abs']
        if early_stop is not None:
            assert early_stop in early_stop_type, 'early stop type must be in {}, bug got {}' \
                .format(early_stop_type, early_stop)

        # communicate suffix
        self.comm_suffix = 'fedavg'

        # check param correctness
        self.check_trainer_param([self.epochs,
                                  self.validation_freq,
                                  self.save_freq,
                                  self.aggregate_every_n_epoch],
                                 ['epochs',
                                  'validation_freq',
                                  'save_freq',
                                  'aggregate_every_n_epoch'],
                                 self.is_pos_int,
                                 '{} is not a positive int')
        self.check_trainer_param([self.secure_aggregate, self.weighted_aggregation, self.pin_memory, self.save_to_local_dir], [
                                 'secure_aggregate', 'weighted_aggregation', 'pin_memory', 'save_to_local_dir'], self.is_bool, '{} is not a bool')
        self.check_trainer_param(
            [self.tol], ['tol'], self.is_float, '{} is not a float')

    def _init_aggregator(self, train_set):
        # compute round to aggregate
        cur_agg_round = 0
        if self.aggregate_every_n_epoch is not None:
            aggregate_round = self.epochs // self.aggregate_every_n_epoch
        else:
            aggregate_round = self.epochs

        # initialize fed avg client
        if self.fed_mode:
            if self.weighted_aggregation:
                sample_num = len(train_set)
            else:
                sample_num = 1.0

            if self._enable_deepspeed and dist.get_rank() == 0:
                client_agg = SecureAggClient(
                    True, aggregate_weight=sample_num, communicate_match_suffix=self.comm_suffix)
            else:
                client_agg = None
        else:
            client_agg = None

        return client_agg, aggregate_round

    def set_model(self, model: t.nn.Module):

        if not issubclass(type(model), t.nn.Module):
            raise ValueError('model must be a subclass of pytorch nn.Module')
        self.model = model
        if self.cuda is not None:
            self.model = self.model.cuda(self.cuda_main_device)
            if self.data_parallel:
                self.parallel_model = DataParallel(model, device_ids=self.cuda, output_device=self.cuda_main_device)

    def _select_model(self):
        if self.data_parallel:
            return self.parallel_model
        else:
            return self.model

    def train_an_epoch(self, epoch_idx, model, train_set, optimizer, loss):

        epoch_loss = 0.0
        batch_idx = 0
        acc_num = 0

        if self.data_loader is None:
            self.data_loader = DataLoader(
                train_set,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                shuffle=self.shuffle,
                num_workers=self.data_loader_worker,
                collate_fn=getattr(train_set, "collate_fn", None)
            )

        dl = self.data_loader

        if not self.fed_mode:
            to_iterate = tqdm.tqdm(dl)
        else:
            to_iterate = dl

        for batch_data, batch_label in to_iterate:

            if self.cuda is not None:
                batch_data, batch_label = self.to_cuda(
                    batch_data, self.cuda_main_device), self.to_cuda(batch_label, self.cuda_main_device)
            elif self._enable_deepspeed:
                batch_data, batch_label = self.to_cuda(
                    batch_data, self.model.device), self.to_cuda(batch_label, self.model.device)

            optimizer.zero_grad()
            pred = model(batch_data)

            if not loss and hasattr(pred, "loss"):
                batch_loss = pred.loss
            elif loss is not None:
                batch_loss = loss(pred, batch_label)
            else:
                raise ValueError(
                    'FedAVGTrainer requires a loss function, but got None, please specify loss function in the'
                    ' job configuration')

            if not self._enable_deepspeed:
                batch_loss.backward()
                optimizer.step()
                batch_loss_np = batch_loss.detach().numpy(
                ) if self.cuda is None else batch_loss.cpu().detach().numpy()
            else:
                batch_loss = batch_loss.mean()
                batch_loss_np = batch_loss.cpu().detach().numpy()
                model.backward(batch_loss)
                model.step()

            if acc_num + self.batch_size > len(train_set):
                batch_len = len(train_set) - acc_num
            else:
                batch_len = self.batch_size
            epoch_loss += batch_loss_np * batch_len
            batch_idx += 1
            LOGGER.debug(f"finish epoch={epoch_idx}, batch={batch_idx}")

        if self.fed_mode:
            LOGGER.debug(
                'epoch {} batch {} finished'.format(epoch_idx, batch_idx))

        epoch_loss = epoch_loss / len(train_set)
        return epoch_loss

    def train(
            self,
            train_set: Dataset,
            validate_set: Dataset = None,
            optimizer: t.optim.Optimizer = None,
            loss=None,
            extra_dict={}):

        if optimizer is None:
            raise ValueError(
                'FedAVGTrainer requires an optimizer, but got None, please specify optimizer in the '
                'job configuration')

        if self.batch_size > len(train_set) or self.batch_size == -1:
            self.batch_size = len(train_set)

        # compute round to aggregate
        cur_agg_round = 0
        client_agg, aggregate_round = self._init_aggregator(train_set)

        # running var
        cur_epoch = 0
        loss_history = []
        need_stop = False
        evaluation_summary = {}

        # training process
        for i in range(self.epochs):

            cur_epoch = i
            LOGGER.info('epoch is {}'.format(i))
            model = self._select_model()
            epoch_loss = self.train_an_epoch(i, model, train_set, optimizer, loss)
            self.callback_loss(epoch_loss, i)
            loss_history.append(float(epoch_loss))
            LOGGER.info('epoch loss is {}'.format(epoch_loss))

            # federation process, if running local mode, cancel federation
            if client_agg is not None or self._enable_deepspeed:
                if not (self.aggregate_every_n_epoch is not None and (i + 1) % self.aggregate_every_n_epoch != 0):

                    # model averaging, only aggregate trainable param
                    if not self._enable_deepspeed or dist.get_rank() == 0:
                        self.model = client_agg.model_aggregation(self.model)
                        if self._enable_deepspeed and dist.get_world_size() > 1:
                            self._share_model()
                    elif self._enable_deepspeed and dist.get_world_size() > 1:
                        self._share_model()

                    # agg loss and get converge status
                    if not self._enable_deepspeed or dist.get_rank() == 0:
                        converge_status = client_agg.loss_aggregation(epoch_loss)
                        cur_agg_round += 1
                        self._sync_converge_status(converge_status)
                    else:
                        converge_status = self._sync_converge_status()

                    LOGGER.info(
                        'model averaging finished, aggregate round {}/{}'.format(
                            cur_agg_round, aggregate_round))
                    if converge_status:
                        LOGGER.info('early stop triggered, stop training')
                        need_stop = True

            # validation process
            if self.validation_freq and ((i + 1) % self.validation_freq == 0):
                LOGGER.info('running validation')
                ids_t, pred_t, label_t = self._predict(train_set)
                evaluation_summary = self.evaluation(
                    ids_t,
                    pred_t,
                    label_t,
                    dataset_type='train',
                    epoch_idx=i,
                    task_type=self.task_type)
                if validate_set is not None:
                    ids_v, pred_v, label_v = self._predict(validate_set)
                    evaluation_summary = self.evaluation(
                        ids_v,
                        pred_v,
                        label_v,
                        dataset_type='validate',
                        epoch_idx=i,
                        task_type=self.task_type)

            # save check point process
            """
            if self.save_freq is not None and ((i + 1) % self.save_freq == 0):

                if self.save_to_local_dir:
                    self.local_checkpoint(
                        self.model, i, optimizer, converge_status=need_stop, loss_history=loss_history)
                else:
                    self.checkpoint(
                        self.model, i, optimizer, converge_status=need_stop, loss_history=loss_history)
                LOGGER.info('save checkpoint : epoch {}'.format(i))
            """

            # if meet stop condition then stop
            if need_stop:
                break

        # post-process
        best_epoch = int(np.array(loss_history).argmin())

        """
        if self.save_to_local_dir:
            self.local_save(model=self.model, optimizer=optimizer, epoch_idx=cur_epoch, loss_history=loss_history,
                            converge_status=need_stop, best_epoch=best_epoch)
        else:
            self.save(model=self.model, optimizer=optimizer, epoch_idx=cur_epoch, loss_history=loss_history,
                      converge_status=need_stop, best_epoch=best_epoch)
        """

        self.summary({
            'best_epoch': best_epoch,
            'loss_history': loss_history,
            'need_stop': need_stop,
            'metrics_summary': evaluation_summary
        })

    def _predict(self, dataset: Dataset):
        pred_result = []

        # switch eval mode
        dataset.eval()
        model = self._select_model()
        model.eval()

        if not dataset.has_sample_ids():
            dataset.init_sid_and_getfunc(prefix=dataset.get_type())

        labels = []
        with torch.no_grad():

            for batch_data, batch_label in DataLoader(
                    dataset, self.batch_size):
                if self.cuda is not None:
                    batch_data = self.to_cuda(batch_data, self.cuda_main_device)
                pred = model(batch_data)
                pred_result.append(pred)
                labels.append(batch_label)

            ret_rs = torch.concat(pred_result, axis=0)
            ret_label = torch.concat(labels, axis=0)

        # switch back to train mode
        dataset.train()
        model.train()

        return dataset.get_sample_ids(), ret_rs, ret_label

    def predict(self, dataset: Dataset):
        return

        ids, ret_rs, ret_label = self._predict(dataset)

        if self.fed_mode:
            return self.format_predict_result(
                ids, ret_rs, ret_label, task_type=self.task_type)
        else:
            return ret_rs, ret_label

    def server_aggregate_procedure(self, extra_data={}):

        # converge status
        check_converge = False
        converge_func = None
        if self.early_stop:
            check_converge = True
            converge_func = converge_func_factory(
                self.early_stop, self.tol).is_converge
            LOGGER.info(
                'check early stop, converge func is {}'.format(converge_func))

        LOGGER.info('server running aggregate procedure')
        server_agg = SecureAggServer(True, communicate_match_suffix=self.comm_suffix)

        # aggregate and broadcast models
        for i in range(self.epochs):
            if not (self.aggregate_every_n_epoch is not None and (i + 1) % self.aggregate_every_n_epoch != 0):

                # model aggregate
                server_agg.model_aggregation()
                converge_status = False

                # loss aggregate
                agg_loss, converge_status = server_agg.loss_aggregation(
                    check_converge=check_converge, converge_func=converge_func)
                self.callback_loss(agg_loss, i)

                # save check point process
                if self.save_freq is not None and ((i + 1) % self.save_freq == 0):
                    self.checkpoint(epoch_idx=i)
                    LOGGER.info('save checkpoint : epoch {}'.format(i))

                # check stop condition
                if converge_status:
                    LOGGER.debug('stop triggered, stop aggregation')
                    break

        LOGGER.info('server aggregation process done')

    def _share_model(self):
        rank = dist.get_rank()

        if rank == 0:
            for p in self.model.parameters():
                if p.requires_grad:
                    scatter_list = [p.data for _ in range(dist.get_world_size())]
                    dist.scatter(p.data, scatter_list, async_op=False)
        else:
            for p in self.model.parameters():
                if p.requires_grad:
                    dist.scatter(p.data, src=0, async_op=False)

        dist.barrier()

    def _sync_converge_status(self, converge_status=None):
        if dist.get_rank() == 0:
            t_status = self.to_cuda(torch.Tensor([converge_status]), self.model.device)
            dist.scatter(t_status, [t_status for _ in range(dist.get_world_size())], async_op=False)
        else:
            t_status = torch.Tensor([False])
            dist.scatter(t_status, src=0, async_op=False)
            return t_status[0].item()
