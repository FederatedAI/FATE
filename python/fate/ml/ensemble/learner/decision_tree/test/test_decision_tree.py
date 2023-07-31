import pandas as pd
from fate.ml.ensemble.learner.decision_tree.hetero.guest import HeteroDecisionTreeGuest
from fate.ml.ensemble.learner.decision_tree.hetero.host import HeteroDecisionTreeHost
from fate.arch.dataframe import PandasReader, DataFrame
import logging
from fate.ml.ensemble.learner.decision_tree.tree_core.loss import BCELoss
from fate.arch import Context
import sys


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
    max_depth = 2
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

        from fate.ml.ensemble.utils.binning import binning
        bin_info = binning(data_guest, max_bin=32)
        bin_data = data_guest.bucketize(boundaries=bin_info)

        loss_bce = BCELoss()
        label = data_guest.label
        init_score = loss_bce.initialize(label)
        predict = loss_bce.predict(init_score)
        empty_gh = data_guest.create_frame()
        loss_bce.compute_grad(empty_gh, label, predict)
        loss_bce.compute_hess(empty_gh, label, predict)

        tree = HeteroDecisionTreeGuest(max_depth)
        ret = tree.booster_fit(ctx, bin_data, empty_gh, bin_info)
        
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

        from fate.ml.ensemble.utils.binning import binning
        bin_info = binning(data_host, max_bin=32)
        bin_data = data_host.bucketize(boundaries=bin_info)

        tree = HeteroDecisionTreeHost(max_depth)
        ret = tree.booster_fit(ctx, bin_data, bin_info)
