import sys

import yaml

from pipeline.backend.pipeline import PipeLine
from pipeline.component.data_statistics import DataStatistics
from pipeline.component.dataio import DataIO
from pipeline.component.hetero_feature_binning import HeteroFeatureBinning
from pipeline.component.hetero_feature_selection import HeteroFeatureSelection
from pipeline.component.hetero_secureboost import HeteroSecureBoost
from pipeline.component.intersection import Intersection
from pipeline.component.psi import PSI
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
        backend = conf.get("backend", 0)
        work_mode = conf.get("work_mode", 0)

    guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
    host_train_data = {"name": "breast_hetero_host", "namespace": "experiment"}

    guest_eval_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
    host_eval_data = {"name": "breast_hetero_host", "namespace": "experiment"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=host)

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
    dataio_0 = DataIO(name="dataio_0")  # start component numbering at 0
    dataio_1 = DataIO(name="dataio_1")

    # get DataIO party instance of guest
    dataio_0_guest_party_instance = dataio_0.get_party_instance(role='guest', party_id=guest)
    # configure DataIO for guest
    dataio_0_guest_party_instance.algorithm_param(with_label=True, output_format="dense")
    # get and configure DataIO party instance of host
    dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")
    intersection_1 = Intersection(name="intersection_1")

    param = {
        "name": 'hetero_feature_binning_0',
        "method": 'optimal',
        "optimal_binning_param": {
            "metric_method": "iv"
        },
        "bin_indexes": -1
    }
    hetero_feature_binning_0 = HeteroFeatureBinning(**param)
    hetero_feature_binning_1 = HeteroFeatureBinning(name='hetero_feature_binning_1')

    statistic_0 = DataStatistics(name='statistic_0')
    psi_0 = PSI(name='psi_0')

    param = {
                "task_type": "classification",
                "learning_rate": 0.1,
                "num_trees": 5,
                "subsample_feature_rate": 1,
                "n_iter_no_change": False,
                "tol": 0.0001,
                "bin_num": 50,
                "objective_param": {
                    "objective": "cross_entropy"
                },
                "encrypt_param": {
                    "method": "paillier"
                },
                "predict_param": {
                    "threshold": 0.5
                },
                "cv_param": {
                    "n_splits": 5,
                    "shuffle": False,
                    "random_seed": 103,
                    "need_cv": False
                },
                "validation_freqs": 1
            }

    secureboost_0 = HeteroSecureBoost(name='secureboost_0', **param)

    param = {
        "name": 'hetero_feature_selection_0',
        "filter_methods": ["manually", "iv_filter", "statistic_filter", "psi_filter", "hetero_sbt_filter"],
        "manually_param": {
            "filter_out_indexes": [1]
        },
        "iv_param": {
            "metrics": ["iv", "iv"],
            "filter_type": ["top_k", "threshold"],
            "take_high": [True, True],
            "threshold": [10, 0.01]
        },
        "statistic_param": {
            "metrics": ["coefficient_of_variance", "skewness"],
            "filter_type": ["threshold", "threshold"],
            "take_high": [True, True],
            "threshold": [0.001, -0.01]
        },
        "psi_param": {
            "metrics": ["psi"],
            "filter_type": ["threshold"],
            "take_high": [False],
            "threshold": [0.1]
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
    # set data input sources of intersection components
    pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(intersection_1, data=Data(data=dataio_1.output.data))
    # set train & validate data of hetero_lr_0 component
    pipeline.add_component(hetero_feature_binning_0, data=Data(data=intersection_0.output.data))
    pipeline.add_component(hetero_feature_binning_1, data=Data(data=intersection_1.output.data),
                           model=Model(hetero_feature_binning_0.output.model))

    pipeline.add_component(statistic_0, data=Data(data=intersection_0.output.data))
    pipeline.add_component(psi_0, data=Data(train_data=intersection_0.output.data,
                                            validate_data=intersection_1.output.data))
    pipeline.add_component(secureboost_0, data=Data(train_data=intersection_0.output.data,
                                                    validate_data=intersection_1.output.data))

    pipeline.add_component(hetero_feature_selection_0, data=Data(data=intersection_0.output.data),
                           model=Model(isometric_model=[hetero_feature_binning_0.output.model,
                                                        statistic_0.output.model,
                                                        psi_0.output.model,
                                                        secureboost_0.output.model]))
    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    pipeline.fit(backend=backend, work_mode=work_mode)
    # query component summary
    print(pipeline.get_component("hetero_feature_selection_0").get_summary())


if __name__ == "__main__":
    main(sys.argv[1])
