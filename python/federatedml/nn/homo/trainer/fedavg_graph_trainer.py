import torch
import torch as t
import numpy as np
from torch_geometric.loader import NeighborLoader
from federatedml.framework.homo.aggregator.secure_aggregator import SecureAggregatorClient as SecureAggClient
from federatedml.nn.dataset.base import Dataset
from federatedml.nn.homo.trainer.fedavg_trainer import FedAVGTrainer
from federatedml.util import LOGGER


class FedAVGGraphTrainer(FedAVGTrainer):
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
    cuda: bool, use cuda or not
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
    """

    def __init__(self, epochs=10, batch_size=512,  # training parameter
                 early_stop=None, tol=0.0001,  # early stop parameters
                 secure_aggregate=True, weighted_aggregation=True, aggregate_every_n_epoch=None,  # federation
                 cuda=False, pin_memory=True, shuffle=True, data_loader_worker=0,  # GPU & dataloader
                 validation_freqs=None,  # validation configuration
                 checkpoint_save_freqs=None,  # checkpoint configuration
                 task_type='auto',
                 num_neighbors=[10,10],
                 ):

        super(FedAVGGraphTrainer, self).__init__(
            epochs=epochs, batch_size=batch_size,  # training parameter
            early_stop=early_stop, tol=tol,  # early stop parameters
            secure_aggregate=secure_aggregate, weighted_aggregation=weighted_aggregation, aggregate_every_n_epoch=aggregate_every_n_epoch,  # federation
            cuda=cuda, pin_memory=pin_memory, shuffle=shuffle, data_loader_worker=data_loader_worker,  # GPU & dataloader
            validation_freqs=validation_freqs,  # validation configuration
            checkpoint_save_freqs=checkpoint_save_freqs,  # checkpoint configuration
            task_type=task_type,
        )
        self.comm_suffix = 'fedavg_graph'
        LOGGER.debug("num_neighbors={}".format(num_neighbors))
        self.num_neighbors = num_neighbors

    def train(
            self,
            train_set: Dataset,
            validate_set: Dataset = None,
            optimizer: t.optim.Optimizer = None,
            loss=None,
            extra_dict={}):

        if self.cuda:
            self.model = self.model.cuda()

        if optimizer is None:
            raise ValueError(
                'FedAVGGraphTrainer requires an optimizer, but got None, please specify optimizer in the '
                'job configuration')
        if loss is None:
            raise ValueError(
                'FedAVGGraphTrainer requires a loss function, but got None, please specify loss function in the'
                ' job configuration')

        if self.batch_size > len(train_set) or self.batch_size == -1:
            self.batch_size = len(train_set)
        dl = NeighborLoader(
            data=train_set.data,
            num_neighbors=self.num_neighbors,
            input_nodes=train_set.input_nodes,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,            
            num_workers=self.data_loader_worker)

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

            client_agg = SecureAggClient(
                True, aggregate_weight=sample_num, communicate_match_suffix=self.comm_suffix)
        else:
            client_agg = None

        # running var
        cur_epoch = 0
        loss_history = []
        need_stop = False
        evaluation_summary = {}

        LOGGER.debug(self.model)

        # training process
        for i in range(self.epochs):
            cur_epoch = i
            LOGGER.info('epoch is {}'.format(i))
            epoch_loss = 0.0
            batch_idx = 0
            acc_num = 0

            for _, batch in enumerate(dl):
                label = batch.y[:self.batch_size]
                optimizer.zero_grad()
                pred = self.model(batch.x, batch.edge_index)[:self.batch_size]
                batch_loss = loss(pred, label)
                batch_loss.backward()
                optimizer.step()
                batch_loss_np = batch_loss.detach().numpy(
                ) if not self.cuda else batch_loss.cpu().detach().numpy()
                if acc_num + self.batch_size > len(train_set):
                    batch_len = len(train_set) - acc_num
                else:
                    batch_len = self.batch_size
                epoch_loss += batch_loss_np * batch_len
                batch_idx += 1

                if self.fed_mode:
                    LOGGER.debug(
                        'epoch {} batch {} finished'.format(
                            i, batch_idx))

            # loss compute
            epoch_loss = epoch_loss / len(train_set)
            self.callback_loss(epoch_loss, i)
            loss_history.append(float(epoch_loss))
            LOGGER.info('epoch loss is {}'.format(epoch_loss))

            # federation process, if running local mode, cancel federation
            if client_agg is not None:
                if not (self.aggregate_every_n_epoch is not None and (i + 1) % self.aggregate_every_n_epoch != 0):
                    # model averaging
                    self.model = client_agg.model_aggregation(self.model)
                    # agg loss and get converge status
                    converge_status = client_agg.loss_aggregation(epoch_loss)
                    cur_agg_round += 1
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
            if self.save_freq is not None and ((i + 1) % self.save_freq == 0):
                self.checkpoint(
                    i, self.model, optimizer, converge_status=need_stop, loss_history=loss_history)
                LOGGER.info('save checkpoint : epoch {}'.format(i))

            # if meet stop condition then stop
            if need_stop:
                break

        # post-process
        best_epoch = int(np.array(loss_history).argmin())
        self.save(model=self.model, optimizer=optimizer, epoch_idx=cur_epoch, loss_history=loss_history,
                  converge_status=need_stop, best_epoch=best_epoch)
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
        self.model.eval()

        if not dataset.has_sample_ids():
            dataset.init_sid_and_getfunc(prefix=dataset.get_type())

        dl = NeighborLoader(
            data=dataset.data,
            num_neighbors=self.num_neighbors,
            input_nodes=dataset.input_nodes,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.data_loader_worker)

        labels = []
        with torch.no_grad():

            for _, batch in enumerate(dl):
                label = batch.y[:self.batch_size]
                pred = self.model(batch.x, batch.edge_index)[:self.batch_size]
                pred_result.append(pred)
                labels.append(label)

            ret_rs = torch.concat(pred_result, axis=0)
            ret_label = torch.concat(labels, axis=0)

        # switch back to train mode
        dataset.train()
        self.model.train()

        LOGGER.debug(dataset.get_sample_ids())
        LOGGER.debug(ret_rs)
        LOGGER.debug(ret_label)
        return dataset.get_sample_ids(), ret_rs, ret_label



