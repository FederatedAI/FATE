import torch as t
from federatedml.util import LOGGER
from federatedml.nn.homo.trainer.trainer_base import TrainerBase
from torch.utils.data import DataLoader
from federatedml.framework.homo.aggregator.secure_aggregator import SecureAggregatorClient



class CustTrainer(TrainerBase):
    
    def __init__(self, epochs, batch_size=256, dataloader_worker=4):
        super(CustTrainer, self).__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataloader_worker = dataloader_worker

    def train(self, dataset, optimizer, loss):
        
        fed_avg = None
        if self.run_local_mode:
            fed_avg = SecureAggregatorClient(max_aggregate_round=epochs, sample_number=len(dataset))
        
        dl = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.dataloader_worker)
        
        for epoch_idx in range(0, self.epochs):
            l_sum = 0
            for data, label in dl:
                optimizer.zero_grad()
                pred = self.model(data)
                l = loss(pred, label)
                l.backward()
                optimizer.step()
                l_sum += l
                
            LOGGER.info('loss sum is {}'.format(l_sum))
            
            if fed_avg:
                fed_avg.aggregate(self.model, l_sum.cpu().detach().numpy())
                
        LOGGER.info('training finished!')
