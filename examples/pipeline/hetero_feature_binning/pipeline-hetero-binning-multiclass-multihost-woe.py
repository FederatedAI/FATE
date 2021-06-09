#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import argparse

from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.component.evaluation import Evaluation
from pipeline.interface.model import Model
from pipeline.component.intersection import Intersection
from pipeline.component.hetero_feature_binning import HeteroFeatureBinning

from pipeline.utils.tools import load_job_config
from pipeline.runtime.entity import JobParameters


def main(config="../../config.yaml", namespace=""):

    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)

    parties = config.parties
    guest = parties.guest[0]
    host = parties.host

    backend = config.backend
    work_mode = config.work_mode

    guest_train_data = {"name": "vehicle_scale_hetero_guest", "namespace": f"experiment{namespace}"}
    guest_validate_data = {"name": "vehicle_scale_hetero_guest", "namespace": f"experiment{namespace}"}

    host_train_data = {"name": "vehicle_scale_hetero_host", "namespace": f"experiment{namespace}"}
    host_validate_data = {"name": "vehicle_scale_hetero_host", "namespace": f"experiment{namespace}"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

    dataio_0, dataio_1 = DataIO(name="dataio_0"), DataIO(name='dataio_1')
    reader_0, reader_1 = Reader(name="reader_0"), Reader(name='reader_1')

    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)
    dataio_0.get_party_instance(role='guest', party_id=guest).component_param(with_label=True, output_format="dense")
    dataio_0.get_party_instance(role='host', party_id=host).component_param(with_label=False, output_format="dense")

    reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=guest_validate_data)
    reader_1.get_party_instance(role='host', party_id=host).component_param(table=host_validate_data)
    dataio_1.get_party_instance(role='guest', party_id=guest).component_param(with_label=True, output_format="dense")
    dataio_1.get_party_instance(role='host', party_id=host).component_param(with_label=True, output_format="dense")

    intersection_0 = Intersection(name="intersection_0")
    intersection_1 = Intersection(name="intersection_1")

    param = {
        "method": "quantile",
        "optimal_binning_param": {
            "metric_method": "gini",
            "min_bin_pct": 0.05,
            "max_bin_pct": 0.8,
            "init_bucket_method": "quantile",
            "init_bin_nums": 100,
            "mixture": True
        },
        "compress_thres": 10000,
        "head_size": 10000,
        "error": 0.001,
        "bin_num": 10,
        "bin_indexes": -1,
        "bin_names": None,
        "category_indexes": [0, 1, 2],
        "category_names": None,
        "adjustment_factor": 0.5,
        "local_only": False,
        "transform_param": {
            "transform_cols": -1,
            "transform_names": None,
            "transform_type": "woe"
        }
    }

    hetero_feature_binning_0 = HeteroFeatureBinning(name="hetero_feature_binning_0", **param)
    hetero_feature_binning_0.get_party_instance(role='host', party_id=host).\
        component_param(transform_param={
            "transform_cols": -1,
            "transform_names": None,
            "transform_type": "bin_num"
        })

    hetero_feature_binning_1 = HeteroFeatureBinning(name='hetero_feature_binning_1')

    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(reader_1)
    pipeline.add_component(dataio_1, data=Data(data=reader_1.output.data), model=Model(dataio_0.output.model))
    pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(intersection_1, data=Data(data=dataio_1.output.data))
    pipeline.add_component(hetero_feature_binning_0, data=Data(data=intersection_0.output.data))
    pipeline.add_component(hetero_feature_binning_1, data=Data(data=intersection_1.output.data),
                           model=Model(hetero_feature_binning_0.output.model))

    pipeline.compile()
    job_parameters = JobParameters(backend=backend, work_mode=work_mode)
    pipeline.fit(job_parameters)

    # predict
    # deploy required components
    pipeline.deploy_component([dataio_0, intersection_0, hetero_feature_binning_0])

    predict_pipeline = PipeLine()
    # add data reader onto predict pipeline
    predict_pipeline.add_component(reader_1)
    # add selected components from train pipeline onto predict pipeline
    # specify data source
    predict_pipeline.add_component(pipeline,
                                   data=Data(predict_input={pipeline.dataio_0.input.data: reader_1.output.data}))
    # run predict model
    predict_pipeline.predict(job_parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
