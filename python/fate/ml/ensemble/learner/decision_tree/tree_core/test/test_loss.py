from fate.arch import Context
from fate.arch.computing.standalone import CSession
from fate.arch.context import Context
from fate.arch.federation.standalone import StandaloneFederation
import pandas as pd
from fate.arch.dataframe import PandasReader
from fate.ml.nn.dataset.table import TableDataset
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
arbiter = ("arbiter", 10000)
guest = ("guest", 10000)
host = ("host", 9999)
name = "fed"

def create_ctx(local):
    from fate.arch import Context
    from fate.arch.computing.standalone import CSession
    from fate.arch.federation.standalone import StandaloneFederation
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    computing = CSession()
    return Context(computing=computing,
                   federation=StandaloneFederation(computing, name, local, [guest, host, arbiter]))

ctx = create_ctx(guest)

df = pd.read_csv(
    '../../../../../../../../examples/data/breast_hetero_guest.csv')
df['sample_id'] = [i for i in range(len(df))]

reader = PandasReader(
    sample_id_name='sample_id',
    match_id_name="id",
    label_name="y",
    dtype="object")

data = reader.to_frame(ctx, df)

