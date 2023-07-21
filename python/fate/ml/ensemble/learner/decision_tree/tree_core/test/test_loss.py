import pandas as pd
from fate.arch.dataframe import PandasReader
import logging
from fate.ml.ensemble.learner.decision_tree.tree_core.loss import BCELoss, L2Loss, CELoss


# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)
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

ctx = create_ctx(guest)

df = pd.read_csv(
    '../../../../../../../../examples/data/breast_hetero_guest.csv')
df['sample_id'] = [i for i in range(len(df))]

df_reg = pd.read_csv('../../../../../../../../examples/data/student_hetero_guest.csv')
df_reg['sample_id'] = [i for i in range(len(df_reg))]

df_multi = pd.read_csv('../../../../../../../../examples/data/student_hetero_guest.csv')
df_multi['sample_id'] = [i for i in range(len(df_multi))]


reader = PandasReader(
    sample_id_name='sample_id',
    match_id_name="id",
    label_name="y",
    dtype="object")

data = reader.to_frame(ctx, df)
data_reg = reader.to_frame(ctx, df_reg)
data_multi = reader.to_frame(ctx, df_multi)

# test loss here
# loss_bce = BCELoss()
# label = data.label
# init_score = loss_bce.initialize(label)
# predict = loss_bce.predict(init_score)
# loss = loss_bce.compute_loss(label, predict)
# empty_gh = data.create_frame()
# loss_bce.compute_grad(empty_gh, label, predict)
# loss_bce.compute_hess(empty_gh, label, predict)

# loss_l2 = L2Loss()
# label = data_reg.label
# init_score = loss_l2.initialize(label)
# predict = loss_l2.predict(init_score)
# loss = loss_l2.compute_loss(label, predict)
# empty_gh = data_reg.create_frame()
# loss_l2.compute_grad(empty_gh, label, predict)
# loss_l2.compute_hess(empty_gh, label, predict)


loss = CELoss()
label = data_multi.label
init_score = loss.initialize(label, class_num=4)
predict = loss.predict(init_score)
# loss = loss.compute_loss(label, predict)
# empty_gh = data_reg.create_frame()
# loss.compute_grad(empty_gh, label, predict)
# loss.compute_hess(empty_gh, label, predict)