import pandas as pd
import numpy as np
from fate.arch.dataframe import PandasReader
import sys
from datetime import datetime


def get_current_datetime_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M")

guest = ("guest", "10000")
host = ("host", "9999")
name = get_current_datetime_str()


def create_ctx(local, context_name):
    from fate.arch import Context
    from fate.arch.computing.backends.standalone import CSession
    from fate.arch.federation.backends.standalone import StandaloneFederation
    import logging

    # prepare log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # init fate context
    computing = CSession()
    return Context(
        computing=computing, federation=StandaloneFederation(computing, context_name, local, [guest, host])
    )


ctx = create_ctx(guest, get_current_datetime_str())
reader = PandasReader(sample_id_name="sample_id", match_id_name="id", dtype="float32")

# set random seed
np.random.seed(42)

# random g h, np array
sample_num = 10000
sample_id = np.array([i for i in range(sample_num)])
id = np.array([i for i in range(sample_num)])
g = np.random.rand(sample_num)
h = np.random.rand(sample_num)
df_gh = pd.DataFrame({"sample_id": sample_id, "id": id, "g": g, "h": h})
data_gh = reader.to_frame(ctx, df_gh)

sample_pos = np.array([0 for i in range(sample_num)])
sample_pos_df = pd.DataFrame({"sample_id": sample_id, "id": id, "node_idx": sample_pos})
sample_pos = reader.to_frame(ctx, sample_pos_df)

from fate.arch.dataframe import DataFrame
from fate.ml.ensemble.utils.sample import goss_sample

rs = goss_sample(data_gh, 0.2, 0.1, random_seed=42)
selected_sample = sample_pos.loc(rs.get_indexer(target='sample_id'))
