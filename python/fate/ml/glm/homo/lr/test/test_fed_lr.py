from fate.ml.nn.algo.homo.fedavg import FedAVGCLient, FedArguments, TrainingArguments, FedAVGServer
import torch as t
import pandas as pd
import sys
from fate.arch.dataframe import PandasReader
from fate.ml.glm.homo.lr.client import HomoLRClient
from fate.ml.glm.homo.lr.server import HomoLRServer


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



if __name__ == "__main__":

    if sys.argv[1] == "guest":

        ctx = create_ctx(guest)
        df = pd.read_csv(
        '../../../../../../../examples/data/breast_homo_guest.csv')
        df['sample_id'] = [i for i in range(len(df))]

        reader = PandasReader(
            sample_id_name='sample_id',
            match_id_name="id",
            label_name="y",
            dtype="object")
        
        data = reader.to_frame(ctx, df)


    elif sys.argv[1] == "host":

        ctx = create_ctx(guest)
        df = pd.read_csv(
        '../../../../../../../examples/data/breast_homo_host.csv')
        df['sample_id'] = [i for i in range(len(df))]

        reader = PandasReader(
            sample_id_name='sample_id',
            match_id_name="id",
            label_name="y",
            dtype="object")

        data = reader.to_frame(ctx, df)

    else:
        ctx = create_ctx(arbiter)