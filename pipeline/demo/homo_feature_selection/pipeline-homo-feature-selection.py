import sys

import yaml

from pipeline.backend.pipeline import PipeLine
from pipeline.component.data_statistics import DataStatistics
from pipeline.component.dataio import DataIO
from pipeline.component.hetero_feature_selection import HeteroFeatureSelection
from pipeline.component.homo_secureboost import HomoSecureBoost
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.interface.model import Model

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def main(config="./config.yaml"):
    """
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", default="./config.yaml", type=str,
                        help="config file")
    args = parser.parse_args()
    file = args.config
    """
    # obtain config
    with open(config, "r") as f:
        conf = yaml.load(f, Loader=Loader)
        parties = conf.get("parties", {})
        if len(parties) == 0:
            raise ValueError(f"Parties id must be sepecified.")
        host = parties["host"][0]
        guest = parties["guest"][0]
        arbiter = parties["arbiter"][0]
        backend = conf.get("backend", 0)
        work_mode = conf.get("work_mode", 0)

    guest_train_data = {"name": "breast_homo_guest", "namespace": "experiment"}
    host_train_data = {"name": "breast_homo_host", "namespace": "experiment"}

    guest_eval_data = {"name": "homo_breast_test", "namespace": "experiment"}
    host_eval_data = {"name": "homo_breast_test", "namespace": "experiment"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_eval_data)
    reader_1.get_party_instance(role='host', party_id=host).algorithm_param(table=host_eval_data)

    # define DataIO components
    dataio_0 = DataIO(name="dataio_0", with_label=True, output_format="dense")  # start component numbering at 0
    dataio_1 = DataIO(name="dataio_1")

    statistic_0 = DataStatistics(name='statistic_0')

    param = {
        "task_type": "classification",
        "learning_rate": 0.1,
        "num_trees": 3,
        "subsample_feature_rate": 1,
        "n_iter_no_change": False,
        "tol": 0.0001,
        "bin_num": 50,
        "validation_freqs": 1,
        "tree_param": {
            "max_depth": 3
        },
        "objective_param": {
            "objective": "cross_entropy"
        },
        "predict_param": {
            "threshold": 0.5
        },
        "cv_param": {
            "n_splits": 5,
            "shuffle": False,
            "random_seed": 103,
            "need_cv": False
        }
    }

    secureboost_0 = HomoSecureBoost(name='secureboost_0', **param)

    param = {
        "name": 'homo_feature_selection_0',
        "filter_methods": ["manually", "homo_sbt_filter"],
        "manually_param": {
            "filter_out_indexes": [1]
        },
        "sbt_param": {
            "metrics": ["feature_importance"],
            "filter_type": ["threshold"],
            "take_high": [True],
            "threshold": [0.1]
        },
        "select_col_indexes": -1
    }
    hetero_feature_selection_0 = HeteroFeatureSelection(**param)

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    # set dataio_1 to replicate model from dataio_0
    pipeline.add_component(dataio_1, data=Data(data=reader_1.output.data), model=Model(dataio_0.output.model))

    # set train & validate data of hetero_lr_0 component

    pipeline.add_component(secureboost_0, data=Data(train_data=dataio_0.output.data,
                                                    validate_data=dataio_1.output.data))

    pipeline.add_component(hetero_feature_selection_0, data=Data(data=dataio_0.output.data),
                           model=Model(isometric_model=[secureboost_0.output.model]))
    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    pipeline.fit(backend=backend, work_mode=work_mode)
    # query component summary
    print(pipeline.get_component("homo_feature_selection_0").get_summary())


if __name__ == "__main__":
    main(sys.argv[1])
