import pandas as pd
from fate.arch.dataframe import PandasReader
import sys
from fate.ml.ensemble.algo.secureboost.hetero.guest import HeteroSecureBoostGuest
from fate.ml.ensemble.algo.secureboost.hetero.host import HeteroSecureBoostHost
from datetime import datetime


def get_current_datetime_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M")

guest = ("guest", "10000")
host_0 = ("host", "9999")
host_1 = ("host", "9998")
name = get_current_datetime_str()


def create_ctx(local, context_name):
    from fate.arch import Context
    from fate.arch.computing.standalone import CSession
    from fate.arch.federation.standalone import StandaloneFederation
    import logging

    # prepare log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # init fate context
    computing = CSession()
    return Context(
        computing=computing, federation=StandaloneFederation(computing, context_name, local, [guest, host_0, host_1])
    )


if __name__ == "__main__":

    party = sys.argv[1]
    max_depth = 3
    num_tree = 1

    if party == "guest":

        ctx = create_ctx(guest, get_current_datetime_str())
        df = pd.read_csv("./../../../../../../../examples/data/breast_hetero_guest.csv")
        df["sample_id"] = [i for i in range(len(df))]

        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", label_name="y", dtype="float32")

        data_guest = reader.to_frame(ctx, df)

        trees = HeteroSecureBoostGuest(num_tree, max_depth=max_depth)
        trees.fit(ctx, data_guest)
        pred = trees.get_train_predict().as_pd_df()

        pred_ = trees.predict(ctx, data_guest)
        
    elif party == "host_0":

        ctx = create_ctx(host_0, get_current_datetime_str())
        df_host = pd.read_csv("~/FATE/FATE-2.0/scripts/hetero_breast_host_0.csv")
        df_host["sample_id"] = [i for i in range(len(df_host))]

        reader_host = PandasReader(sample_id_name="sample_id", match_id_name="id", dtype="float32")

        data_host = reader_host.to_frame(ctx, df_host)

        trees = HeteroSecureBoostHost(num_tree, max_depth=max_depth)
        trees.fit(ctx, data_host)

        trees.predict(ctx, data_host)

    elif party == "host_1":

        ctx = create_ctx(host_1, get_current_datetime_str())
        df_host = pd.read_csv("~/FATE/FATE-2.0/scripts/hetero_breast_host_1.csv")
        df_host["sample_id"] = [i for i in range(len(df_host))]

        reader_host = PandasReader(sample_id_name="sample_id", match_id_name="id", dtype="float32")

        data_host = reader_host.to_frame(ctx, df_host)

        trees = HeteroSecureBoostHost(num_tree, max_depth=max_depth)
        trees.fit(ctx, data_host)

        trees.predict(ctx, data_host)