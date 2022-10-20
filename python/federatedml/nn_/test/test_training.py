import torch as t
from federatedml.nn_.model_zoo.cwjnet import TestNet
from federatedml.nn_.model_zoo.LR import LogisticRegression
from federatedml.nn_.dataset.table import TableDataset
from federatedml.nn_.homo.trainer.trainer_base import get_trainer_class

t.manual_seed(100)

dataset = TableDataset(None, feature_dtype='float', label_dtype='float')
dataset.load('/home/cwj/standalone_fate_install_1.9.0_release/examples/data/breast_homo_guest.csv')
network = LogisticRegression(31)
trainer_class = get_trainer_class('fedavg_trainer')
trainer = trainer_class()
trainer.set_model(network)
trainer.local_mode()
trainer.train(dataset, batch_size=128, optimizer=t.optim.Adam(network.parameters(), lr=0.01), loss=t.nn.BCELoss(),
              epochs=30, torch_seed=100)
pred_rs = trainer.predict(dataset)
