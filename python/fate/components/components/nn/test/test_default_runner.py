from fate.components.components.nn.runner.default_runner import DefaultRunner
from fate_client.pipeline.components.fate.nn.fate_torch import nn, optim
from fate_client.pipeline.components.fate.nn.fate_torch.base import Sequential
from fate_client.pipeline.components.fate.homo_nn import get_config_of_default_runner
from fate_client.pipeline.components.fate.nn.loader import DatasetLoader
from fate_client.pipeline.components.fate.nn.algo_params import TrainingArguments, FedAVGArguments
from fate.arch import Context
from fate.arch.computing.standalone import CSession
from fate.arch.context import Context
from fate.arch.federation.standalone import StandaloneFederation
import pandas as pd
from fate.arch.dataframe import PandasReader
import logging
from fate.components.core import GUEST

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


computing = CSession()
ctx = Context("guest", computing=computing, federation=StandaloneFederation(
    computing, "fed", ("guest", 10000), [("host", 9999)]), )

df = pd.read_csv(
    './../../../../../../examples/data/vehicle_scale_homo_guest.csv')
df['sample_id'] = [i for i in range(len(df))]

reader = PandasReader(
    sample_id_name='sample_id',
    match_id_name="id",
    label_name="y",
    dtype="object")
data = reader.to_frame(ctx, df)

runner_conf = get_config_of_default_runner(
    algo='fedavg',
    model=Sequential(
        nn.Linear(
            18,
            10),
        nn.ReLU(),
        nn.Linear(
            10,
            4),
        nn.Softmax()),
    loss=nn.CrossEntropyLoss(),
    dataset=DatasetLoader(
        'table',
        'TableDataset',
        flatten_label=True,
        label_dtype='long'),
    optimizer=optim.Adam(
        lr=0.01),
    training_args=TrainingArguments(
        num_train_epochs=50,
        per_device_train_batch_size=128),
    fed_args=FedAVGArguments(),
    task_type='binary')

runner = DefaultRunner(**runner_conf)
runner.set_context(ctx)
runner.set_role(GUEST)
runner.local_mode = True
runner.train(data)
rs = runner.predict(data)
