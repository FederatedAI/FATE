from fate.arch import Context
from fate.arch.computing.standalone import CSession
from fate.arch.context import Context
from fate.arch.federation.standalone import StandaloneFederation
import pandas as pd
from fate.arch.dataframe import PandasReader
from fate.ml.nn.dataset.table import TableDataset
from fate.ml.glm.homo.lr.client import HomoLRClient
import logging

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
    computing, "fed", ("guest", 10000), [("guest", 10000), ("host", 9999)]) )

df = pd.read_csv(
    '../../../../../../../examples/data/breast_homo_guest.csv')
df['sample_id'] = [i for i in range(len(df))]

reader = PandasReader(
    sample_id_name='sample_id',
    match_id_name="id",
    label_name="y",
    dtype="object")
reader_2 = PandasReader(
    sample_id_name='sample_id',
    match_id_name="id",
    dtype="object")
data = reader.to_frame(ctx, df)

# df = data.as_pd_df()
data_2 = reader_2.to_frame(ctx, df.drop(columns=['y']))
ds = TableDataset(return_dict=True, to_tensor=True)
ds.load(data)


client = HomoLRClient(
    50, 800, optimizer_param={
        'method': 'adam', 'penalty': 'l1', 'aplha': 0.1, 'optimizer_para': {
            'lr': 0.1}}, init_param={
                'method': 'random', 'fill_val': 1.0}, learning_rate_scheduler={
                    'method': 'linear', 'scheduler_params': {'start_factor'}})
client.l2 = 0.01
client.l1 = 0.01
client.local_mode = True
client.fit(ctx, data, validate_data=data)
export_model = client.get_model()
pred = client.predict(ctx, data)
# pred_2 = client.predict(ctx, data_2)
