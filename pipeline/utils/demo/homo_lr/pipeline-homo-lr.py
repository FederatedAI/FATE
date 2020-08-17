import sys

import yaml

from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.homo_lr import HomoLR
from pipeline.component.evaluation import Evaluation
from pipeline.component.reader import Reader
from pipeline.interface.data import Data

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

    # define DataIO components
    dataio_0 = DataIO(name="dataio_0", with_label=True, output_format="dense")  # start component numbering at 0

    param = {
        "penalty": "L2",
        "validation_freqs": 1,
        "early_stopping_rounds": None,
        "max_iter": 10,
        "cv_param": {
            "need_cv": False,
            "n_splits": 3,
            "shuffle": True,
            "random_seed": 13
        }
    }

    homo_lr_0 = HomoLR(name='homo_lr_0', **param)

    evaluation_0 = Evaluation(name='evaluation_0')
    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    # set data input sources of intersection components

    pipeline.add_component(homo_lr_0, data=Data(train_data=dataio_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=homo_lr_0.output.data))
    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    pipeline.fit(backend=backend, work_mode=work_mode)
    # query component summary
    print(pipeline.get_component("homo_lr_0").get_summary())
    print(pipeline.get_component("evaluation_0").get_summary())


if __name__ == "__main__":
    main(sys.argv[1])
