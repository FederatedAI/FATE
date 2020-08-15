import sys

import yaml

from pipeline.backend.pipeline import PipeLine
from pipeline.component.data_statistics import DataStatistics
from pipeline.component.dataio import DataIO
from pipeline.component.intersection import Intersection
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
        hosts = parties["host"][0]
        guest = parties["guest"][0]
        arbiter = parties["arbiter"][0]
        backend = conf.get("backend", 0)
        work_mode = conf.get("work_mode", 0)

    guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
    host_train_data = {"name": "breast_hetero_host", "namespace": "experiment"}

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=hosts).algorithm_param(table=host_train_data)

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest). \
        set_roles(guest=guest, host=hosts, arbiter=arbiter)
    dataio_0 = DataIO(name="dataio_0")

    dataio_0.get_party_instance(role='guest', party_id=guest). \
        algorithm_param(with_label=True, output_format="dense")
    dataio_0.get_party_instance(role='host', party_id=hosts).algorithm_param(with_label=False)

    dataio_1 = DataIO(name="dataio_1")

    intersect_0 = Intersection(name="intersection_0")
    intersect_0.algorithm_param(intersect_method="rsa")
    intersect_1 = Intersection(name="intersection_1")
    statics_0 = DataStatistics(name="data_statistic_0")

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(dataio_1, model=Model(model=dataio_0.output.model),
                           data=Data(data=reader_0.output.data))
    pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(intersect_1, data=Data(data=dataio_1.output.data))
    pipeline.add_component(statics_0, data=Data(data=intersect_0.output.data))

    pipeline.compile()

    pipeline.fit(backend=backend, work_mode=work_mode)

    print(f'intersection data output: {pipeline.get_component("intersection_0").get_output_data()}')
    print(f'data_statistic data output: {pipeline.get_component("data_statistic_0").get_output_data()}')
    print(f'summary: {pipeline.get_component("data_statistic_0").get_summary()}')


if __name__ == "__main__":
    main(sys.argv[1])
