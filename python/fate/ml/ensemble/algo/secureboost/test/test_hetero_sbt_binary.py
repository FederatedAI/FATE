import pandas as pd
from fate.arch.dataframe import PandasReader
import sys
from fate.ml.ensemble.algo.secureboost.hetero.guest import HeteroSecureBoostGuest
from fate.ml.ensemble.algo.secureboost.hetero.host import HeteroSecureBoostHost
from datetime import datetime

def get_current_datetime_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M")

guest = ("guest", "10000")
host = ("host", "9999")
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
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # init fate context
    computing = CSession()
    return Context(
        computing=computing, federation=StandaloneFederation(computing, context_name, local, [guest, host])
    )


if __name__ == "__main__":

    party = sys.argv[1]
    max_depth = 3
    num_tree = 3

    if party == "guest":

        ctx = create_ctx(guest, get_current_datetime_str())
        df = pd.read_csv("./../../../../../../../examples/data/breast_hetero_guest.csv")
        df["sample_id"] = [i for i in range(len(df))]

        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", label_name="y", dtype="float32")
        data_guest = reader.to_frame(ctx, df)
        trees = HeteroSecureBoostGuest(num_tree, max_depth=max_depth, l1=1.0)
        trees.fit(ctx, data_guest)
        pred = trees.get_train_predict().as_pd_df()
        pred_ = trees.predict(ctx, data_guest).as_pd_df()

        # compute auc
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(pred["label"], pred["predict_score"])
        print(auc)

    elif party == "host":

        ctx = create_ctx(host, get_current_datetime_str())
        df_host = pd.read_csv("./../../../../../../../examples/data/breast_hetero_host.csv")
        df_host["sample_id"] = [i for i in range(len(df_host))]

        reader_host = PandasReader(sample_id_name="sample_id", match_id_name="id", dtype="float32")

        data_host = reader_host.to_frame(ctx, df_host)

        trees = HeteroSecureBoostHost(num_tree, max_depth=max_depth)
        trees.fit(ctx, data_host)
        trees.predict(ctx, data_host)
