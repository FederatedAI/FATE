import pandas as pd

from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.arch.dataframe import PandasReader
from fate.arch.launchers.multiprocess_launcher import launch
from fate.ml.ensemble.algo.secureboost.hetero.guest import HeteroSecureBoostGuest
from fate.ml.ensemble.algo.secureboost.hetero.host import HeteroSecureBoostHost


def train(ctx: Context, data: DataFrame, num_trees: int = 3, objective: str = 'binary:bce', max_depth: int = 3,
          learning_rate: float = 0.3):
    if ctx.is_on_guest:
        bst = HeteroSecureBoostGuest(num_trees=num_trees, objective=objective, \
                                     max_depth=max_depth, learning_rate=learning_rate)
    else:
        bst = HeteroSecureBoostHost(num_trees=num_trees, max_depth=max_depth)

    bst.fit(ctx, data)

    return bst


def predict(ctx: Context, data: DataFrame, model_dict: dict):
    if ctx.is_on_guest:
        bst = HeteroSecureBoostGuest()
    else:
        bst = HeteroSecureBoostHost()
    bst.from_model(model_dict)
    return bst.predict(ctx, data)


def csv_to_df(ctx, file_path, has_label=True):
    df = pd.read_csv(file_path)
    df["sample_id"] = [i for i in range(len(df))]
    if has_label:
        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", label_name="y", dtype="float32")
    else:
        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", dtype="float32")

    fate_df = reader.to_frame(ctx, df)
    return fate_df


def run(ctx):
    num_tree = 3
    max_depth = 3
    if ctx.is_on_guest:
        data = csv_to_df(ctx, '../data/breast_hetero_guest.csv')
        bst = train(ctx, data, num_trees=num_tree, max_depth=max_depth)
        model_dict = bst.get_model()
        pred = predict(ctx, data, model_dict)
        print(pred.as_pd_df())
    else:
        data = csv_to_df(ctx, '../data/breast_hetero_host.csv', has_label=False)
        bst = train(ctx, data, num_trees=num_tree, max_depth=max_depth)
        model_dict = bst.get_model()
        predict(ctx, data, model_dict)


if __name__ == '__main__':
    launch(run)
