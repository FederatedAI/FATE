import importlib

import click

from fate.arch.launchers.multiprocess_launcher import launch


def run_pearson(ctx):
    from fate.ml.statistics.pearson_correlation import PearsonCorrelation
    from fate.arch import dataframe

    ctx.mpc.init()

    inst = PearsonCorrelation()
    kwargs = {
        "sample_id_name": None,
        "match_id_name": "id",
        "delimiter": ",",
        "label_name": "y",
        "label_type": "float32",
        "dtype": "float32",
    }
    input_data = dataframe.CSVReader(**kwargs).to_frame(ctx, "/Users/sage/FATE-SBT/examples/data/breast_hetero_guest.csv")
    inst.fit(ctx, input_data=input_data)


if __name__ == "__main__":
    launch(run_pearson)
