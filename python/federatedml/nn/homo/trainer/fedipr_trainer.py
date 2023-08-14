import torch as t
import tqdm
import numpy as np
import torch
from typing import Literal
from federatedml.nn.homo.trainer.fedavg_trainer import FedAVGTrainer
from federatedml.nn.backend.utils import distributed_util
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from federatedml.nn.dataset.watermark import WaterMarkImageDataset, WaterMarkDataset
from federatedml.util import LOGGER
from federatedml.nn.model_zoo.sign_block import generate_signature, is_sign_block
from federatedml.nn.model_zoo.sign_block import SignatureBlock
from sklearn.metrics import accuracy_score
from federatedml.nn.dataset.base import Dataset
from federatedml.util import consts


def get_sign_blocks(model: torch.nn.Module):
    
    record_sign_block = {}
    for name, m in model.named_modules():
        if is_sign_block(m):
            record_sign_block[name] = m

    return record_sign_block


def get_keys(sign_block_dict: dict, num_bits: int):
    
    key_pairs = {}  
    param_len = [] 
    sum_allocated_bits = 0
    # Iterate through each layer and compute the flattened parameter lengths
    for k, v in sign_block_dict.items():
        param_len.append(len(v.embeded_param.flatten()))
    total_param_len = sum(param_len)

    alloc_bits = {}

    for i, (k, v) in enumerate(sign_block_dict.items()):
        allocated_bits = int((param_len[i] / total_param_len) * num_bits)
        alloc_bits[k] = allocated_bits
        sum_allocated_bits += allocated_bits

    rest_bits = num_bits - sum_allocated_bits
    if rest_bits > 0:
        alloc_bits[k] += rest_bits

    for k, v in sign_block_dict.items():
        key_pairs[k] = generate_signature(v, alloc_bits[k])
    
    return key_pairs


"""
Verify Tools
"""

def to_cuda(var, device=0):
    if hasattr(var, 'cuda'):
        return var.cuda(device)
    elif isinstance(var, tuple) or isinstance(var, list):
        ret = tuple(to_cuda(i) for i in var)
        return ret
    elif isinstance(var, dict):
        for k in var:
            if hasattr(var[k], 'cuda'):
                var[k] = var[k].cuda(device)
        return var
    else:
        return var


def _verify_sign_blocks(sign_blocks, keys, cuda=False, device=None):

    signature_correct_count = 0
    total_bit = 0
    for name, block in sign_blocks.items():
        block: SignatureBlock = block
        W, signature = keys[name]
        if cuda:
            W = to_cuda(W, device=device)
            signature = to_cuda(signature, device=device)
        extract_bits = block.extract_sign(W)
        total_bit += len(extract_bits)
        signature_correct_count += (extract_bits == signature).sum().detach().cpu().item()
    
    sign_acc = signature_correct_count / total_bit
    return sign_acc


def _suggest_sign_bit(param_num, client_num):
    max_signbit = param_num // client_num
    max_signbit -= 1  # not to exceed
    if max_signbit <= 0:
        raise ValueError('not able to add feature based watermark, param_num is {}, client num is {}, computed max bit is {} <=0'.format(param_num, client_num, max_signbit))
    return max_signbit


def compute_sign_bit(model, client_num):
    total_param_num = 0
    blocks = get_sign_blocks(model)
    for k, v in blocks.items():
        total_param_num += v.embeded_param_num()
    if total_param_num == 0:
        return 0
    return _suggest_sign_bit(total_param_num, client_num)


def verify_feature_based_signature(model, keys):
    
    model = model.cpu()
    sign_blocks = get_sign_blocks(model)
    return _verify_sign_blocks(sign_blocks, keys, cuda=False)



class FedIPRTrainer(FedAVGTrainer):

    def __init__(self, epochs=10,  noraml_dataset_batch_size=32, watermark_dataset_batch_size=2,
                  early_stop=None, tol=0.0001, secure_aggregate=True, weighted_aggregation=True, 
                 aggregate_every_n_epoch=None, cuda=None, pin_memory=True, shuffle=True, 
                 data_loader_worker=0, validation_freqs=None, checkpoint_save_freqs=None, 
                 task_type='auto', save_to_local_dir=False, collate_fn=None, collate_fn_params=None,
                 alpha=0.01, verify_freqs=1, backdoor_verify_method: Literal['accuracy', 'loss'] = 'accuracy'
                 ):
        
        super().__init__(epochs, noraml_dataset_batch_size, early_stop, tol, secure_aggregate, weighted_aggregation, 
                         aggregate_every_n_epoch, cuda, pin_memory, shuffle, data_loader_worker, 
                         validation_freqs, checkpoint_save_freqs, task_type, save_to_local_dir, collate_fn, collate_fn_params)
        
        self.normal_train_set = None
        self.watermark_set = None
        self.data_loader = None
        self.normal_dataset_batch_size = noraml_dataset_batch_size
        self.watermark_dataset_batch_size = watermark_dataset_batch_size
        self.alpha = alpha
        self.verify_freqs = verify_freqs
        self.backdoor_verify_method = backdoor_verify_method
        self._sign_keys = None
        self._sign_blocks = None
        self._client_num = None
        self._sign_bits = None

        assert self.alpha > 0, 'alpha must be greater than 0'
        assert self.verify_freqs > 0 and isinstance(self.verify_freqs, int), 'verify_freqs must be greater than 0'
        assert self.backdoor_verify_method in ['accuracy', 'loss'], 'backdoor_verify_method must be accuracy or loss'

    def local_mode(self):
        self.fed_mode = False
        self._client_num = 1

    def _handle_dataset(self, train_set, collate_fn):

        if not distributed_util.is_distributed() or distributed_util.get_num_workers() <= 1:
            return DataLoader(
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
            return DataLoader(
                train_set,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                num_workers=self.data_loader_worker,
                collate_fn=collate_fn,
                sampler=train_sampler
            )

    
    def _get_train_data_loader(self, train_set):

        collate_fn = self._get_collate_fn(train_set)

        if isinstance(train_set, WaterMarkDataset):
            LOGGER.info('detect watermark dataset, split watermark dataset and normal dataset')
            normal_train_set = train_set.get_normal_dataset()
            watermark_set = train_set.get_watermark_dataset()
            if normal_train_set is None:
                raise ValueError('normal dataset must not be None in FedIPR algo')
            train_dataloder = self._handle_dataset(normal_train_set, collate_fn)

            if watermark_set is not None:
                watermark_dataloader = self._handle_dataset(watermark_set, collate_fn)
            else:
                watermark_dataloader = None
            self.normal_train_set = normal_train_set
            self.watermark_set = watermark_set
            dataloaders = {'train': train_dataloder, 'watermark': watermark_dataloader}
            return dataloaders
        else:
            LOGGER.info('detect non-watermark dataset')
            train_dataloder = self._handle_dataset(train_set, collate_fn)
            dataloaders = {'train': train_dataloder, 'watermark': None}
            return dataloaders

    def _get_device(self):
        if self.cuda is not None or self._enable_deepspeed:
            device = self.cuda_main_device if self.cuda_main_device is not None else self.model.device
            return device
        else:
            return None
    
    def verify(self, sign_blocks: dict, keys: dict):

        return _verify_sign_blocks(sign_blocks, keys, self.cuda is not None, self._get_device())

    def get_loss_from_pred(self, loss, pred, batch_label):

        if not loss and hasattr(pred, "loss"):
            batch_loss = pred.loss

        elif loss is not None:
            if batch_label is None:
                raise ValueError(
                    "When loss is set, please provide label to calculate loss"
                )
            if not isinstance(pred, torch.Tensor) and hasattr(pred, "logits"):
                pred = pred.logits
            batch_loss = loss(pred, batch_label)
        else:
            raise ValueError(
                'Trainer requires a loss function, but got None, please specify loss function in the'
                ' job configuration')
        
        return batch_loss
    
    def _get_keys(self, sign_blocks):
        
        if self._sign_keys is None:
            self._sign_keys = get_keys(sign_blocks, self._sign_bits)
        return self._sign_keys
    
    def _get_sign_blocks(self):
        if self._sign_blocks is None:
            sign_blocks = get_sign_blocks(self.model)
            self._sign_blocks = sign_blocks

        return self._sign_blocks
    
    def train(self, train_set: Dataset, validate_set: Dataset = None, optimizer = None, loss=None, extra_dict={}):
        
        if 'keys' in extra_dict:
            self._sign_keys = extra_dict['keys']
            self._sign_bits = extra_dict['num_bits']
        else:
            LOGGER.info('computing feature based sign bits')
            if self._client_num is None and self.party_id_list is not None:
                self._client_num = len(self.party_id_list)
            self._sign_bits = compute_sign_bit(self.model, self._client_num)
        

        LOGGER.info('client num {}, party id list {}'.format(self._client_num, self.party_id_list))
        LOGGER.info('will assign {} bits for feature based watermark'.format(self._sign_bits))
        return super().train(train_set, validate_set, optimizer, loss, extra_dict)

    def train_an_epoch(self, epoch_idx, model, train_set, optimizer, loss_func):

        epoch_loss = 0.0
        batch_idx = 0
        acc_num = 0

        sign_blocks = self._get_sign_blocks()
        keys = self._get_keys(sign_blocks)

        dl, watermark_dl = self.data_loader['train'], self.data_loader['watermark']
        if isinstance(dl, DistributedSampler):
            dl.sampler.set_epoch(epoch_idx)
        if isinstance(watermark_dl, DistributedSampler):
            watermark_dl.sampler.set_epoch(epoch_idx)

        if not self.fed_mode:
            trainset_iterator = tqdm.tqdm(dl)
        else:
            trainset_iterator = dl
        batch_label = None

        # collect watermark data and mix them into the training data
        watermark_collect = []
        if watermark_dl is not None:
            for watermark_batch in watermark_dl:
                watermark_collect.append(watermark_batch)

        for _batch_iter in trainset_iterator:

            _batch_iter = self._decode(_batch_iter)

            if isinstance(_batch_iter, list) or isinstance(_batch_iter, tuple):
                batch_data, batch_label = _batch_iter
            else:
                batch_data = _batch_iter

            if watermark_dl is not None:
                # Mix the backdoor sample into the training data
                wm_batch_idx = int(batch_idx % len(watermark_collect))
                wm_batch = watermark_collect[wm_batch_idx]
                if isinstance(wm_batch, list):
                    wm_batch_data, wm_batch_label = wm_batch
                    batch_data = torch.cat([batch_data, wm_batch_data], dim=0)
                    batch_label = torch.cat([batch_label, wm_batch_label], dim=0)
                else:
                    wm_batch_data = wm_batch
                    batch_data = torch.cat([batch_data, wm_batch_data], dim=0)

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

            sign_loss = 0
            # Get the sign loss of model
            for name, block in sign_blocks.items():

                block: SignatureBlock = block
                W, signature = keys[name]
                if self.cuda is not None:
                    device = self._get_device()
                    W = self.to_cuda(W, device)
                    signature = self.to_cuda(signature, device)
                sign_loss += self.alpha * block.sign_loss(W, signature)

            
            batch_loss = self.get_loss_from_pred(loss_func, pred, batch_label)     
            batch_loss += sign_loss

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

            if self.fed_mode:
                LOGGER.debug(
                    'epoch {} batch {} finished'.format(epoch_idx, batch_idx))

        epoch_loss = epoch_loss / len(train_set)

        # verify the sign of model during training
        if epoch_idx % self.verify_freqs == 0:
            # verify feature-based signature
            sign_acc = self.verify(sign_blocks, keys)
            LOGGER.info(f"epoch {epoch_idx} sign accuracy: {sign_acc}")
            # verify backdoor signature
            if self.watermark_set is not None:
                _, pred, label = self._predict(self.watermark_set)
                pred = pred.detach().cpu()
                label = label.detach().cpu()
                if self.backdoor_verify_method == 'accuracy':
                    if not isinstance(pred, torch.Tensor) and hasattr(pred, "logits"):
                        pred = pred.logits
                    pred = pred.numpy().reshape((len(label), -1))
                    label = label.numpy()
                    pred_label = np.argmax(pred, axis=1)
                    metric = accuracy_score(pred_label.flatten(), label.flatten())
                else:
                    metric = self.get_loss_from_pred(loss_func, pred, label)

                LOGGER.info(f"epoch {epoch_idx} backdoor {self.backdoor_verify_method}: {metric}")

        return epoch_loss

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
        
        if isinstance(dataset, WaterMarkDataset):
            normal_train_set = dataset.get_normal_dataset()
            if normal_train_set is None:
                raise ValueError('normal train set is None in FedIPR algo predict function')
        else:
            normal_train_set = normal_train_set

        ids, ret_rs, ret_label = self._predict(normal_train_set)

        if self.fed_mode:
            return self.format_predict_result(
                ids, ret_rs, ret_label, task_type=self.task_type)
        else:
            return ret_rs, ret_label
        
    def save(
            self,
            model=None,
            epoch_idx=-1,
            optimizer=None,
            converge_status=False,
            loss_history=None,
            best_epoch=-1,
            extra_data={}):

        extra_data = {'keys': self._sign_keys, 'num_bits': self._sign_bits}
        super().save(model, epoch_idx, optimizer, converge_status, loss_history, best_epoch, extra_data)

    def local_save(self,
                   model=None,
                   epoch_idx=-1,
                   optimizer=None,
                   converge_status=False,
                   loss_history=None,
                   best_epoch=-1,
                   extra_data={}):

        extra_data = {'keys': self._sign_keys, 'num_bits': self._sign_bits}
        super().local_save(model, epoch_idx, optimizer, converge_status, loss_history, best_epoch, extra_data)


