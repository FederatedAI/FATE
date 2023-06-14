from typing import Optional, Union
import torch as t
from fate.ml.nn.algo.homo.fedavg import FedAVGArguments, TrainingArguments, FedAVGCLient, FedAVGServer
from fate.components.components.nn.nn_runner import NNInput, NNOutput, NNRunner
from torch.utils.data import TensorDataset


class MyRunner(NNRunner):

    def __init__(self, in_feat=30, epoch=10, learning_rate=0.01, batch_size=32) -> None:
        super().__init__()
        self.in_feat = in_feat
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def setup(self, df=None):
        
        ctx = self.get_context()

        if self.is_client():

            df = df.drop(columns=['id', 'sample_id'])
            X = df.drop(columns=['y']).values
            y = df['y'].values
            X_tensor = t.tensor(X, dtype=t.float32)
            y_tensor = t.tensor(y, dtype=t.float32)
            dataset = TensorDataset(X_tensor, y_tensor)
            loss_fn = t.nn.BCELoss()

            model = t.nn.Sequential(
                t.nn.Linear(self.in_feat, 10),
                t.nn.ReLU(),
                t.nn.Linear(10, 1),
                t.nn.Sigmoid()
            )

            optimizer = t.optim.Adam(model.parameters(), lr=self.learning_rate)

            train_arg = TrainingArguments(num_train_epochs=self.epoch, per_device_train_batch_size=self.batch_size, disable_tqdm=False)

            fed_arg = FedAVGArguments()

            return FedAVGCLient(ctx=ctx, model=model, optimizer=optimizer, 
                                training_args=train_arg, fed_args=fed_arg, train_set=dataset, loss_fn=loss_fn), dataset

        elif self.is_server():
            return FedAVGServer(ctx=ctx)
    
    def train(self, input_data: NNInput = None):
        if self.is_client():
            df = input_data.get('train_data')
            trainer, _ = self.setup(df)
        elif self.is_server():
            trainer = self.setup()

        trainer.train()
    
    def predict(self, input_data: NNInput = None):

        if self.is_client():
            df = input_data.get('test_data')
            trainer, ds = self.setup(df)
            trainer.set_local_mode()
            pred_rs = trainer.predict(ds)
            print('pred rs is {}'.format(pred_rs))
