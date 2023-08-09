import pandas as pd
from fate.ml.ensemble.learner.decision_tree.hetero.guest import HeteroDecisionTreeGuest
from fate.arch.dataframe import PandasReader, DataFrame
import logging
from fate.ml.ensemble.learner.decision_tree.tree_core.loss import BCELoss
from fate.arch import Context


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

    ctx = create_ctx(guest)

    df = pd.read_csv(
        './../../../../../../../examples/data/breast_hetero_guest.csv')
    df['sample_id'] = [i for i in range(len(df))]

    reader = PandasReader(
        sample_id_name='sample_id',
        match_id_name="id",
        label_name="y",
        dtype="float32")

    data = reader.to_frame(ctx, df)

    from fate.ml.ensemble.utils.binning import binning
    a = binning(data, max_bin=32)
    bin_data = data.bucketize(boundaries=a)

    loss_bce = BCELoss()
    label = data.label
    init_score = loss_bce.initialize(label)
    predict = loss_bce.predict(init_score)
    empty_gh = data.create_frame()
    loss_bce.compute_grad(empty_gh, label, predict)
    loss_bce.compute_hess(empty_gh, label, predict)


    tree = HeteroDecisionTreeGuest(3)
    ret = tree.booster_fit(ctx, bin_data, empty_gh, a)