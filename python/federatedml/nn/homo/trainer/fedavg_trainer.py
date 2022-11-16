import torch
import torch as t
import tqdm
from torch.utils.data import DataLoader
from federatedml.nn.dataset.base import Dataset
from federatedml.nn.homo.trainer.trainer_base import TrainerBase
from federatedml.util import LOGGER, consts
from federatedml.framework.homo.aggregator.secure_aggregator import SecureAggregatorClient


class FedAVGTrainer(TrainerBase):

    """

    Parameters
    ----------
    epochs: int >0, epochs to train
    batch_size: int, -1 means full batch
    early_stop: None, 'diff' or 'abs'. if None, disable early stop; if 'diff', use the loss difference between
                two epochs as early stop condition, if differences < eps, stop training ; if 'abs', if loss < eps,
                stop training
    eps: float, eps value for early stop
    secure_aggregate: bool, default is True, whether to use secure aggregation. if enabled, will add random number
                            mask to local models. These random number masks will eventually cancel out to get 0.
    weighted_aggregation: bool, whether add weight to each local model when doing aggregation.
                         if True, According to origin paper, weight of a client is: n_local / n_global, where n_local
                         is the sample number locally and n_global is the sample number of all clients.
                         if False, simply averaging these models.

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
                 early_stop=None, eps=0.0001,  # early stop parameters
                 secure_aggregate=True, weighted_aggregation=True, aggregate_every_n_epoch=None,  # federation
                 cuda=False, pin_memory=True, shuffle=True, data_loader_worker=0,  # GPU dataloader
                 validation_freqs=None,  # validation configuration
                 checkpoint_save_freqs=None,
                 task_type='auto'
                 ):

        super(FedAVGTrainer, self).__init__()

        # training parameters
        self.epochs = epochs
        self.eps = eps
        self.validation_freq = validation_freqs
        self.save_freq = checkpoint_save_freqs

        self.task_type = task_type
        task_type_allow = [consts.BINARY, consts.REGRESSION, consts.MULTY, 'auto']
        assert self.task_type in task_type_allow, 'task type must in {}'.format(task_type_allow)

        # aggregation param
        self.secure_aggregate = secure_aggregate
        self.weighted_aggregation = weighted_aggregation
        self.aggregate_every_n_epoch = aggregate_every_n_epoch

        # GPU
        self.cuda = cuda
        if not torch.cuda.is_available() and self.cuda:
            raise ValueError('Cuda is not available on this machine')

        # data loader
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.data_loader_worker = data_loader_worker

        self.early_stop = early_stop
        early_stop_type = ['diff', 'abs']
        if early_stop is not None:
            assert early_stop in early_stop_type, 'early stop type must be in {}, bug got {}' \
                .format(early_stop_type, early_stop)

        # check param correctness
        self.check_trainer_param([self.epochs, self.validation_freq, self.save_freq, self.aggregate_every_n_epoch],
                                 ['epochs', 'validation_freq', 'save_freq', 'aggregate_every_n_epoch'], self.is_pos_int,
                                 '{} is not a positive int')
        self.check_trainer_param([self.secure_aggregate, self.weighted_aggregation],
                                 ['secure_aggregate', 'weighted_aggregation'], self.is_bool, '{} is not a bool')
        self.check_trainer_param([self.eps], ['eps'], self.is_float, '{} is not a float')

    def to_cuda(self, var):
        if hasattr(var, 'cuda'):
            return var.cuda()
        elif isinstance(var, tuple) or isinstance(var, list):
            return tuple(self.to_cuda(i) for i in var)
        else:
            return var

    def train(self, train_set: Dataset, validate_set: Dataset = None, optimizer: t.optim.Optimizer = None, loss=None):

        if self.cuda:
            self.model = self.model.cuda()

        if optimizer is None:
            raise ValueError('FedAVGTrainer requires an optimizer, but got None, please specify optimizer in the '
                             'job configuration')
        if loss is None:
            raise ValueError('FedAVGTrainer requires a loss function, but got None, please specify loss function in the'
                             ' job configuration')

        if self.batch_size > len(train_set):
            self.batch_size = len(train_set)
        dl = DataLoader(train_set, batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=self.shuffle,
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
                sample_num = None
            fedavg = SecureAggregatorClient(max_aggregate_round=aggregate_round, secure_aggregate=self.secure_aggregate,
                                            check_convergence=self.early_stop is not None, aggregate_type='fedavg',
                                            eps=self.eps,
                                            convergence_type=self.early_stop, sample_number=sample_num)
        else:
            fedavg = None

        # training process
        for i in range(self.epochs):
            LOGGER.info('epoch is {}'.format(i))
            epoch_loss = 0.0
            batch_idx = 0
            acc_num = 0
            # for better user interface
            if self.fed_mode:
                to_iterate = tqdm.tqdm(dl)
            else:
                to_iterate = dl
            
            for batch_data, batch_label in to_iterate:
                
                if self.cuda:
                    batch_data, batch_label = self.to_cuda(batch_data), self.to_cuda(batch_label)
                    
                optimizer.zero_grad()
                pred = self.model(batch_data)
                batch_loss = loss(pred, batch_label)
                batch_loss.backward()
                optimizer.step()
                batch_loss_np = batch_loss.detach().numpy() if not self.cuda else batch_loss.cpu().detach().numpy()
                if acc_num + self.batch_size > len(train_set):
                    batch_len = len(train_set) - acc_num
                else:
                    batch_len = self.batch_size
                epoch_loss += batch_loss_np * batch_len
                batch_idx += 1

                if self.fed_mode:
                    LOGGER.debug('epoch {} batch {} finished'.format(i, batch_idx))

            epoch_loss = epoch_loss / len(train_set)
            self.callback_loss(epoch_loss, i)
            LOGGER.info('epoch loss is {}'.format(epoch_loss))
            
            # federation process, if running local mode, cancel federation
            if fedavg is not None:
                if self.aggregate_every_n_epoch is not None and (i+1) % self.aggregate_every_n_epoch != 0:
                    continue

                # model averaging
                agg_model, converge_status = fedavg.aggregate(self.model, epoch_loss)

                cur_agg_round += 1
                LOGGER.info('model averaging finished, aggregate round {}/{}'.format(cur_agg_round,
                                                                                     fedavg.max_aggregate_round))
                if converge_status:
                    LOGGER.info('early stop triggered, stop training')
                    break
                    
            if self.validation_freq and ((i+1) % self.validation_freq == 0):
                LOGGER.info('running validation')
                ids_t, pred_t, label_t = self._predict(train_set)
                self.evaluation(ids_t, pred_t, label_t, dataset_type='train', epoch_idx=i, task_type=self.task_type)
                if validate_set is not None:
                    ids_v, pred_v, label_v = self._predict(validate_set)
                    self.evaluation(ids_v, pred_v, label_v, dataset_type='validate', epoch_idx=i, task_type=self.task_type)

            if self.save_freq is not None and ((i+1) % self.save_freq == 0):
                self.set_checkpoint(self.model, optimizer, i)
                LOGGER.info('save checkpoint : epoch {}'.format(i))

        self.export_model(model=self.model, optimizer=optimizer, epoch_idx=self.epochs)

    def _predict(self, dataset: Dataset):

        pred_result = []

        if not dataset.has_sample_ids():
            dataset.generate_sample_ids(prefix=dataset.get_type())

        labels = []
        with torch.no_grad():

            # switch to eval model
            self.model.eval()

            for batch_data, batch_label in DataLoader(dataset, self.batch_size):
                if self.cuda:
                    batch_data = self.to_cuda(batch_label)
                pred = self.model(batch_data)
                pred_result.append(pred)
                labels.append(batch_label)

            self.model.train()

            ret_rs = torch.concat(pred_result, axis=0)
            ret_label = torch.concat(labels, axis=0)

        return dataset.get_sample_ids(), ret_rs, ret_label

    def predict(self, dataset: Dataset):
        ids, ret_rs, ret_label = self._predict(dataset)
        if self.fed_mode:
            return self.format_predict_result(ids, ret_rs, ret_label, task_type=self.task_type)
        else:
            return ret_rs, ret_label