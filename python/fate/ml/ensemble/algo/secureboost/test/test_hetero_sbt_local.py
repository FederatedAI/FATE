import pandas as pd
from fate.arch.dataframe import PandasReader, DataFrame
from fate.arch import Context
import sys
from fate.ml.ensemble.algo.secureboost.hetero.guest import HeteroSecureBoostGuest
from fate.ml.ensemble.algo.secureboost.hetero.host import HeteroSecureBoostHost


arbiter = ("arbiter", "10000")
guest = ("guest", "10000")
host = ("host", "9999")
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

if __name__ == '__main__':

    party = sys.argv[1]
    max_depth = 1
    num_tree = 1
    from sklearn.metrics import roc_auc_score as auc
    if party == 'guest':
        ctx = create_ctx(guest)

        df = pd.read_csv(
            './../../../../../../../examples/data/breast_hetero_guest.csv')
        df['sample_id'] = [i for i in range(len(df))]

        reader = PandasReader(
            sample_id_name='sample_id',
            match_id_name="id",
            label_name="y",
            dtype="float32")
        
        data_guest = reader.to_frame(ctx, df)

        trees = HeteroSecureBoostGuest(num_tree, max_depth=max_depth)
        trees.fit(ctx, data_guest)
        pred = trees.get_cache_predict_score().as_pd_df()
        pred['sample_id'] = pred.sample_id.astype(int)
        df = pd.merge(df, pred, on='sample_id')
        print(auc(df.y, df.predict))

        
    elif party == 'host':
        ctx = create_ctx(host)

        df_host = pd.read_csv(
            './../../../../../../../examples/data/breast_hetero_host.csv')
        df_host['sample_id'] = [i for i in range(len(df_host))]

        reader_host = PandasReader(
            sample_id_name='sample_id',
            match_id_name="id",
            dtype="float32")
        
        data_host = reader_host.to_frame(ctx, df_host)

        trees = HeteroSecureBoostHost(num_tree, max_depth=max_depth)
        trees.fit(ctx, data_host)

