import tqdm
import numpy as np
import torch
from typing import Literal
from federatedml.nn.homo.trainer.fedavg_trainer import FedAVGTrainer
from federatedml.nn.backend.utils import distributed_util
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from federatedml.nn.dataset.watermark import WaterMarkImageDataset
from federatedml.util import LOGGER
from federatedml.nn.model_zoo.ipr.sign_block import generate_signature, is_sign_block
from federatedml.nn.model_zoo.ipr.sign_block import SignatureBlock
from sklearn.metrics import accuracy_score


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




class FedIPRTrainer(FedAVGTrainer):

    def __init__(self, epochs=10,  noraml_dataset_batch_size=32, watermark_dataset_batch_size=2,
                  early_stop=None, tol=0.0001, secure_aggregate=True, weighted_aggregation=True, 
                 aggregate_every_n_epoch=None, cuda=None, pin_memory=True, shuffle=True, 
                 data_loader_worker=0, validation_freqs=None, checkpoint_save_freqs=None, 
                 task_type='auto', save_to_local_dir=False, collate_fn=None, collate_fn_params=None,
                 sign_bits=50, alpha=0.01, verify_freqs=1, backdoor_verify_method: Literal['accuracy', 'loss'] = 'accuracy'
                 ):
        
        super().__init__(epochs, noraml_dataset_batch_size, early_stop, tol, secure_aggregate, weighted_aggregation, 
                         aggregate_every_n_epoch, cuda, pin_memory, shuffle, data_loader_worker, 
                         validation_freqs, checkpoint_save_freqs, task_type, save_to_local_dir, collate_fn, collate_fn_params)
        
        self.normal_train_set = None
        self.watermark_set = None
        self.data_loader = None
        self.sign_bits = sign_bits
        self.normal_dataset_batch_size = noraml_dataset_batch_size
        self.watermark_dataset_batch_size = watermark_dataset_batch_size
        self.alpha = alpha
        self.verify_freqs = verify_freqs
        self.backdoor_verify_method = backdoor_verify_method

        assert self.sign_bits > 0, 'sign_bits must be greater than 0'
        assert self.alpha > 0, 'alpha must be greater than 0'
        assert self.verify_freqs > 0 and isinstance(self.verify_freqs, int), 'verify_freqs must be greater than 0'
        assert self.backdoor_verify_method in ['accuracy', 'loss'], 'backdoor_verify_method must be accuracy or loss'

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

        if not isinstance(train_set, WaterMarkImageDataset):
            raise ValueError('dataset must be a watermark dataset in FedIPR algo')
        
        normal_train_set = train_set.get_normal_dataset()
        watermark_set = train_set.get_watermark_dataset()
        train_dataloder = self._handle_dataset(normal_train_set, collate_fn)
        if watermark_set is not None:
            watermark_dataloader = self._handle_dataset(watermark_set, collate_fn)
        else:
            watermark_dataloader = None
        self.normal_train_set = normal_train_set
        self.watermark_set = watermark_set
        dataloaders = {'train': train_dataloder, 'watermark': watermark_dataloader}
        return dataloaders
    

    def _get_device(self):
        if self.cuda is not None or self._enable_deepspeed:
            device = self.cuda_main_device if self.cuda_main_device is not None else self.model.device
            return device
        else:
            return None
    
    def verify(self, sign_blocks: dict, keys: dict):

        signature_correct_count = 0
        for name, block in sign_blocks.items():
            block: SignatureBlock = block
            W, signature = keys[name]
            if self.cuda is not None:
                device = self._get_device()
                W = self.to_cuda(W, device=device)
                signature = self.to_cuda(signature, device=device)
            extract_bits = block.extract_sign(W)
            signature_correct_count += (extract_bits == signature).sum().detach().cpu().item()
        
        sign_acc = signature_correct_count / self.sign_bits

        return sign_acc

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

    def train_an_epoch(self, epoch_idx, model, train_set, optimizer, loss):

        epoch_loss = 0.0
        batch_idx = 0
        acc_num = 0

        sign_blocks = get_sign_blocks(self.model)
        keys = get_keys(sign_blocks, self.sign_bits)

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

            if isinstance(_batch_iter, list):
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

            batch_loss = self.get_loss_from_pred(loss, pred, batch_label)            
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
                    print(pred_label)
                    print(label)
                    metric = accuracy_score(pred_label.flatten(), label.flatten())
                else:
                    metric = self.get_loss_from_pred(loss, pred, label)

                LOGGER.info(f"epoch {epoch_idx} backdoor {self.backdoor_verify_method}: {metric}")

        return epoch_loss
