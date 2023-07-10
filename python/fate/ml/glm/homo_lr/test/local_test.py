from fate.arch import Context
from fate.arch.computing.standalone import CSession
from fate.arch.context import Context
from fate.arch.federation.standalone import StandaloneFederation
import pandas as pd
from fate.arch.dataframe import PandasReader
from fate.ml.glm.homo_lr.client import HomoLRClient, HomoLRModel



computing = CSession()
ctx = Context(
    "guest",
    computing=computing,
    federation=StandaloneFederation(computing, "fed", ("guest", 10000), [("host", 9999)]),
)

df = pd.read_csv('./examples/data/vehicle_scale_homo_guest.csv')
df['sample_id'] = [i for i in range(len(df))]

reader = PandasReader(sample_id_name='sample_id', match_id_name="id", label_name="y", dtype="object")
data = reader.to_frame(ctx, df)
df = data.as_pd_df()

client = HomoLRClient(50, 32, learning_rate_param=0.01)
a = client.fit(ctx, data)

