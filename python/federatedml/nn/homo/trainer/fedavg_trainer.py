import torch
import torch as t
import torch.distributed as dist
import tqdm
import numpy as np
import transformers
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from federatedml.framework.homo.aggregator.secure_aggregator import SecureAggregatorClient as SecureAggClient
from federatedml.framework.homo.aggregator.secure_aggregator import SecureAggregatorServer as SecureAggServer
from federatedml.nn.backend.utils import deepspeed_util
from federatedml.nn.backend.utils import distributed_util
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
    task_type: str, 'auto', 'binary', 'multi', 'regression',
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
                 save_to_local_dir=False,  # save model to local path
                 collate_fn=None,
                 collate_fn_params=None
                 ):

        super(FedAVGTrainer, self).__init__()

        # training parameters
        self.epochs = epochs
        self.tol = tol
        self.validation_freq = validation_freqs
        self.save_freq = checkpoint_save_freqs
        self.save_to_local_dir = save_to_local_dir

        self.task_type = task_type.lower()
        task_type_allow = [
            consts.BINARY,
            consts.REGRESSION,
            consts.MULTY,
            consts.CAUSAL_LM,
            consts.SEQ_2_SEQ_LM,
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

        self.collate_fn = collate_fn
        self.collate_fn_params = collate_fn_params if collate_fn_params is not None else dict()

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

        # federation
        self.client_agg = None
        self.server_agg = None
        self.aggregate_round = None

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

            if not distributed_util.is_distributed() or distributed_util.is_rank_0():
                client_agg = SecureAggClient(
                    self.secure_aggregate, aggregate_weight=sample_num, communicate_match_suffix=self.comm_suffix)
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
                LOGGER.info('device ids are {}'.format(self.cuda))
                self.parallel_model = DataParallel(model, device_ids=self.cuda, output_device=self.cuda_main_device)

    def _select_model(self):
        if self.data_parallel:
            return self.parallel_model
        else:
            return self.model

    def train_an_epoch(self, epoch_idx, model, train_set, optimizer, loss_func):

        epoch_loss = 0.0
        batch_idx = 0
        acc_num = 0

        if isinstance(self.data_loader.sampler, DistributedSampler):
            self.data_loader.sampler.set_epoch(epoch_idx)

        dl = self.data_loader

        total_batch_len = len(dl)
        LOGGER.info('total batch len is {}'.format(total_batch_len))
        
        if not self.fed_mode:
            to_iterate = tqdm.tqdm(dl)
        else:
            to_iterate = dl

        batch_label = None
        for _batch_iter in to_iterate:
            _batch_iter = self._decode(_batch_iter)
            if isinstance(_batch_iter, list) or isinstance(_batch_iter, tuple):
                batch_data, batch_label = _batch_iter
            else:
                batch_data = _batch_iter

            if self.cuda is not None or self._enable_deepspeed:
                device = self.cuda_main_device if self.cuda_main_device is not None else self.model.device
                batch_data = self.to_cuda(batch_data, device)
                if batch_label is not None:
                    batch_label = self.to_cuda(batch_label, device)

            if not self._enable_deepspeed:
                optimizer.zero_grad()
            else:
                model.zero_grad()

            pred = model(batch_data)

            if not loss_func and hasattr(pred, "loss"):

                if isinstance(model, DataParallel):
                    batch_loss = pred.loss.mean()
                else:
                    batch_loss = pred.loss
                    
            elif loss_func is not None:
                if batch_label is None:
                    raise ValueError(
                        "When loss is set, please provide label to calculate loss"
                    )
                if not isinstance(pred, torch.Tensor) and hasattr(pred, "logits"):
                    pred = pred.logits
                batch_loss = loss_func(pred, batch_label)
            else:
                raise ValueError(
                    'FedAVGTrainer requires a loss function, but got None, please specify loss function in the'
                    ' job configuration')

            if not self._enable_deepspeed:
                batch_loss.backward()
                optimizer.step()
                batch_loss_np = np.array(batch_loss.detach().tolist()) if self.cuda is None \
                    else np.array(batch_loss.cpu().detach().tolist())

                if acc_num + self.batch_size > len(train_set):
                    batch_len = len(train_set) - acc_num
                else:
                    batch_len = self.batch_size

                epoch_loss += batch_loss_np * batch_len
            else:
                batch_loss = model.backward(batch_loss)
                batch_loss_np = np.array(batch_loss.cpu().detach().tolist())
                model.step()
                batch_loss_np = self._sync_loss(batch_loss_np * self._get_batch_size(batch_data))
                if distributed_util.is_rank_0():
                    epoch_loss += batch_loss_np

            batch_idx += 1

            # LOGGER.info(f"finish epoch={epoch_idx}, batch={batch_idx}")
            if self.fed_mode:
                if batch_idx % (total_batch_len // 100) == 0:
                    percentage = (batch_idx / total_batch_len) * 100
                    LOGGER.debug(f"Training progress of epoch {epoch_idx}: {percentage:.1f}%")
                    
        epoch_loss = epoch_loss / len(train_set)
        return epoch_loss

    def on_loop_begin_client(self, **kwargs):
        pass

    def on_loop_end_client(self, **kwargs):
        pass

    def on_loop_begin_server(self, **kwargs):
        pass

    def on_loop_end_server(self, **kwargs):
        pass

    def _client_sends_data(self, epoch_idx, epoch_loss, cur_agg_round):
        need_stop = False
        if self.client_agg is not None or distributed_util.is_distributed():
            if not (self.aggregate_every_n_epoch is not None and (epoch_idx + 1) % self.aggregate_every_n_epoch != 0):

                # model averaging, only aggregate trainable param
                if self._deepspeed_zero_3:
                    deepspeed_util.gather_model(self.model)

                if not distributed_util.is_distributed() or distributed_util.is_rank_0():
                    self.model = self.client_agg.model_aggregation(self.model)
                    if distributed_util.is_distributed() and distributed_util.get_num_workers() > 1:
                        self._share_model()
                else:
                    self._share_model()

                # agg loss and get converge status
                if not distributed_util.is_distributed() or distributed_util.is_rank_0():
                    converge_status = self.client_agg.loss_aggregation(epoch_loss)
                    cur_agg_round += 1
                    if distributed_util.is_distributed() and distributed_util.get_num_workers() > 1:
                        self._sync_converge_status(converge_status)
                else:
                    converge_status = self._sync_converge_status()

                if not distributed_util.is_distributed() or distributed_util.is_rank_0():
                    LOGGER.info(
                        'model averaging finished, aggregate round {}/{}'.format(
                            cur_agg_round, self.aggregate_round))

                if converge_status:
                    LOGGER.info('early stop triggered, stop training')
                    need_stop = True

        return need_stop

    def _server_aggregates_data(self, epoch_idx, check_converge, converge_func):

        need_stop = False
        if not (self.aggregate_every_n_epoch is not None and (epoch_idx + 1) % self.aggregate_every_n_epoch != 0):

            # model aggregate
            self.server_agg.model_aggregation()

            # loss aggregate
            agg_loss, converge_status = self.server_agg.loss_aggregation(
                check_converge=check_converge, converge_func=converge_func)
            self.callback_loss(agg_loss, epoch_idx)

            # save check point process
            if self.save_freq is not None and ((epoch_idx + 1) % self.save_freq == 0):
                self.checkpoint(epoch_idx=epoch_idx)
                LOGGER.info('save checkpoint : epoch {}'.format(epoch_idx))

            # check stop condition
            if converge_status:
                LOGGER.debug('stop triggered, stop aggregation')
                need_stop = True

        return need_stop

    def train(
            self,
            train_set: Dataset,
            validate_set: Dataset = None,
            optimizer: t.optim.Optimizer = None,
            loss=None,
            extra_dict={}):

        if optimizer is None:
            raise ValueError(
                'An optimizer is required, but got None, please specify optimizer in the '
                'job configuration')

        self._optimizer = optimizer
        self._loss_fn = loss

        if self.batch_size > len(train_set) or self.batch_size == -1:
            self.batch_size = len(train_set)

        # compute round to aggregate
        cur_agg_round = 0
        self.client_agg, self.aggregate_round = self._init_aggregator(train_set)

        # running var
        cur_epoch = 0
        loss_history = []
        need_stop = False
        evaluation_summary = {}

        self.data_loader = self._get_train_data_loader(train_set)

        self.on_loop_begin_client()

        # training process
        for i in range(self.epochs):

            cur_epoch = i
            LOGGER.info('epoch is {}'.format(i))
            model = self._select_model()
            epoch_loss = self.train_an_epoch(i, model, train_set, self._optimizer, self._loss_fn)
            if not distributed_util.is_distributed() or distributed_util.is_rank_0():
                self.callback_loss(epoch_loss, i)
                loss_history.append(float(epoch_loss))

            # federation process, if running local mode, cancel federation
            need_stop = self._client_sends_data(i, epoch_loss, cur_agg_round)
            cur_agg_round += 1

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
            if self.save_freq is not None and ((i + 1) % self.save_freq == 0):
                if self._deepspeed_zero_3:
                    deepspeed_util.gather_model(self.model)

            if not distributed_util.is_distributed() or distributed_util.is_rank_0():
                if self.save_freq is not None and ((i + 1) % self.save_freq == 0):

                    if self.save_to_local_dir:
                        self.local_checkpoint(
                            self.model, i, self._optimizer, converge_status=need_stop, loss_history=loss_history)
                    else:
                        self.checkpoint(
                            self.model, i, self._optimizer, converge_status=need_stop, loss_history=loss_history)
                    LOGGER.info('save checkpoint : epoch {}'.format(i))

            # if meet stop condition then stop
            if need_stop:
                break

        # post-process
        if self._deepspeed_zero_3:
            deepspeed_util.gather_model(self.model)

        self.on_loop_end_client()

        if not distributed_util.is_distributed() or distributed_util.is_rank_0():
            best_epoch = int(np.array(loss_history).argmin())

            if self.save_to_local_dir:
                self.local_save(model=self.model, optimizer=self._optimizer, epoch_idx=cur_epoch, loss_history=loss_history,
                                converge_status=need_stop, best_epoch=best_epoch)
            else:
                self.save(model=self.model, optimizer=self._optimizer, epoch_idx=cur_epoch, loss_history=loss_history,
                          converge_status=need_stop, best_epoch=best_epoch)

            best_epoch = int(np.array(loss_history).argmin())
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
            for _batch_iter in DataLoader(
                dataset, self.batch_size
            ):
                if isinstance(_batch_iter, list):
                    batch_data, batch_label = _batch_iter
                else:
                    batch_label = _batch_iter.pop("labels")
                    batch_data = _batch_iter

                if self.cuda is not None or self._enable_deepspeed:
                    device = self.cuda_main_device if self.cuda_main_device is not None else self.model.device
                    batch_data = self.to_cuda(batch_data, device)

                pred = model(batch_data)

                if not isinstance(pred, torch.Tensor) and hasattr(pred, "logits"):
                    pred = pred.logits

                pred_result.append(pred)
                labels.append(batch_label)

            ret_rs = torch.concat(pred_result, axis=0)
            ret_label = torch.concat(labels, axis=0)

        # switch back to train mode
        dataset.train()
        model.train()

        return dataset.get_sample_ids(), ret_rs, ret_label

    def predict(self, dataset: Dataset):
        if self.task_type in [consts.CAUSAL_LM, consts.SEQ_2_SEQ_LM]:
            LOGGER.warning(f"Not support prediction of task_types={[consts.CAUSAL_LM, consts.SEQ_2_SEQ_LM]}")
            return

        if distributed_util.is_distributed() and not distributed_util.is_rank_0():
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
        self.server_agg = SecureAggServer(self.secure_aggregate, communicate_match_suffix=self.comm_suffix)

        self.on_loop_begin_server()
        # aggregate and broadcast models
        for i in range(self.epochs):

            need_stop = self._server_aggregates_data(i, check_converge, converge_func)
            if need_stop:
                break

        self.on_loop_end_server()
        if self._model is not None:
            if self.save_to_local_dir:
                self.local_save(model=self.model, epoch_idx=i, converge_status=need_stop)
            else:
                self.save(model=self.model, epoch_idx=i, converge_status=need_stop)
            LOGGER.info('sever side model saved')

    def _decode(self, data):
        if isinstance(data, transformers.tokenization_utils_base.BatchEncoding):
            return dict(data)
        else:
            return data

    def _get_batch_size(self, data):
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            if "input_ids" in data:
                return data["input_ids"].shape[0]
            else:
                for _, value in data.items():
                    if hasattr(value, "shape"):
                        return value.shape[0]

        raise ValueError("cat not infer batch size from data")

    def _get_collate_fn(self, dataset):
        if not self.collate_fn and not hasattr(dataset, "collate_fn"):
            return None
        if self.collate_fn:
            if not hasattr(dataset, "tokenizer"):
                raise ValueError(f"Collate Fn Only Support in task types=[{consts.CAUSAL_LM}, {consts.SEQ_2_SEQ_LM}]")
            collate_fn = getattr(transformers, self.collate_fn)(dataset.tokenizer, **self.collate_fn_params)
            return collate_fn
        else:
            return dataset.collate_fn

    def _get_train_data_loader(self, train_set):
        collate_fn = self._get_collate_fn(train_set)

        if not distributed_util.is_distributed() or distributed_util.get_num_workers() <= 1:
            data_loader = DataLoader(
                train_set,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                shuffle=self.shuffle,
                num_workers=self.data_loader_worker,
                collate_fn=collate_fn
            )
        else:
            train_sampler = DistributedSampler(
                train_set,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank()
            )
            data_loader = DataLoader(
                train_set,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                num_workers=self.data_loader_worker,
                collate_fn=collate_fn,
                sampler=train_sampler
            )

        return data_loader


    def _share_model(self, sync_trainable_only=True):
      
        if distributed_util.is_rank_0():

            for p in self.model.parameters():
                if (not sync_trainable_only) or (sync_trainable_only and p.requires_grad):
                    scatter_list = [p.data for _ in range(distributed_util.get_num_workers())]
                    dist.scatter(p.data, scatter_list, async_op=False)
        else:

            for p in self.model.parameters():
                if (not sync_trainable_only) or (sync_trainable_only and p.requires_grad):
                    dist.scatter(p.data, src=0, async_op=False)

    def _sync_converge_status(self, converge_status=None):
        if distributed_util.is_rank_0():
            t_status = self.to_cuda(torch.Tensor([converge_status]), self.model.device)
            dist.scatter(t_status, [t_status for _ in range(distributed_util.get_num_workers())], async_op=False)
        else:
            t_status = self.to_cuda(torch.Tensor([False]), self.model.device)
            dist.scatter(t_status, src=0, async_op=False)
            return t_status[0].item()

    def _sync_loss(self, loss):
        if distributed_util.get_num_workers() == 1:
            return loss

        loss = self.to_cuda(torch.tensor(loss), self.model.device)
        if distributed_util.is_rank_0():
            loss_list = [torch.zeros_like(loss) for _ in range(distributed_util.get_num_workers())]
            dist.gather(loss, gather_list=loss_list, async_op=False)
            loss_sum = 0
            for _l in loss_list:
                loss_sum += _l.item()
            return loss_sum
        else:
            dist.gather(loss, dst=0, async_op=False)
            # LOGGER.info(f"Loss on rank{dist.get_rank()}={loss}")
