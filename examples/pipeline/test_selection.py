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
import argparse
import json

from fate_client.pipeline import StandalonePipeline, FateFlowPipeline
from fate_client.pipeline.components.fate import FeatureScale
from fate_client.pipeline.components.fate import Intersection
from fate_client.pipeline.components.fate import Reader
from fate_client.pipeline.components.fate import Statistics, HeteroFeatureSelection
from fate_client.pipeline.utils import test_utils


def main(config="./config.yaml", namespace=""):
    if isinstance(config, str):
        config = test_utils.load_job_config(config)

    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]

    if config.work_mode == 0:
        pipeline = StandalonePipeline().set_roles(guest=guest, host=host, arbiter=arbiter)
    else:
        pipeline = FateFlowPipeline().set_roles(guest=guest, host=host, arbiter=arbiter)
    reader_0 = Reader(name="reader_0")
    cluster = config.work_mode

    if cluster:
        reader_0.guest.component_param(table_name="breast_hetero_guest",
                                       namespace=f"{namespace}experiment",
                                       # path="file:///data/projects/fate/examples/data/breast_hetero_guest.csv",
                                       # format="csv",
                                       # match_id_name="id",
                                       # delimiter=",",
                                       label_name="y",
                                       label_type="float32",
                                       dtype="float32")

        reader_0.hosts[0].component_param(table_name="breast_hetero_host",
                                          namespace=f"{namespace}experiment",
                                          # path="file:///data/projects/fate/examples/data/breast_hetero_host.csv",
                                          # match_id_name="id",
                                          # delimiter=",",
                                          label_name=None,
                                          dtype="float32")
    else:
        data_base = config.data_base_dir

        reader_0.guest.component_param(path=f"file://{data_base}/examples/data/breast_hetero_guest.csv",
                                       # path="file:///data/projects/fate/examples/data/breast_hetero_guest.csv",
                                       format="csv",
                                       match_id_name="id",
                                       delimiter=",",
                                       label_name="y",
                                       label_type="float32",
                                       dtype="float32")

        reader_0.hosts[0].component_param(path=f"file://{data_base}/examples/data/breast_hetero_host.csv",
                                          # path="file:///data/projects/fate/examples/data/breast_hetero_host.csv",
                                          format="csv",
                                          match_id_name="id",
                                          delimiter=",",
                                          label_name=None,
                                          dtype="float32")

    intersection_0 = Intersection(name="intersection_0",
                                  method="raw",
                                  input_data=reader_0.outputs["output_data"])

    intersection_1 = Intersection(name="intersection_1",
                                  method="raw",
                                  input_data=reader_0.outputs["output_data"])

    feature_scale_0 = FeatureScale(name="feature_scale_0",
                                   method="standard",
                                   train_data=intersection_0.outputs["output_data"])

    feature_scale_1 = FeatureScale(name="feature_scale_1",
                                   test_data=intersection_1.outputs["output_data"],
                                   input_model=feature_scale_0.outputs["output_model"])

    statistics_0 = Statistics(name="statistics_0", train_data=feature_scale_1.outputs["test_output_data"],
                              metrics=["mean", "max", "std", "var", "kurtosis", "skewness"])

    selection_0 = HeteroFeatureSelection(name="selection_0", train_data=intersection_0.outputs["output_data"],
                                         method=["statistic"],
                                         input_statistic_model=statistics_0.outputs["output_model"],
                                         statistic_param={"metrics": ["mean", "max", "kurtosis", "skewness"]})

    pipeline.add_task(reader_0)
    pipeline.add_task(feature_scale_0)
    pipeline.add_task(feature_scale_1)
    pipeline.add_task(intersection_0)
    pipeline.add_task(intersection_1)
    pipeline.add_task(statistics_0)
    pipeline.add_task(selection_0)
    pipeline.compile()
    print(pipeline.get_dag())
    pipeline.fit()
    print(json.dumps(pipeline.get_task_info("statistics_0").get_output_model(), indent=4))

    print(json.dumps(pipeline.get_task_info("selection_0").get_output_model(), indent=4))

    predict_pipeline = StandalonePipeline()
    reader_1 = Reader(name="reader_1")
    if cluster:
        reader_1.guest.component_param(table_name="breast_hetero_guest",
                                       namespace=f"{namespace}experiment",
                                       # path="file:///data/projects/fate/examples/data/breast_hetero_guest.csv",
                                       # format="csv",
                                       # match_id_name="id",
                                       # delimiter=",",
                                       label_name="y",
                                       label_type="float32",
                                       dtype="float32")

        reader_1.hosts[0].component_param(table_name="breast_hetero_host",
                                          namespace=f"{namespace}experiment",
                                          # path="file:///data/projects/fate/examples/data/breast_hetero_host.csv",
                                          # match_id_name="id",
                                          # delimiter=",",
                                          label_name=None,
                                          dtype="float32")
    else:
        data_base = config.data_base_dir

        reader_1.guest.component_param(path=f"file://{data_base}/examples/data/breast_hetero_guest.csv",
                                       # path="file:///data/projects/fate/examples/data/breast_hetero_guest.csv",
                                       format="csv",
                                       match_id_name="id",
                                       delimiter=",",
                                       label_name="y",
                                       label_type="float32",
                                       dtype="float32")

        reader_1.hosts[0].component_param(path=f"file://{data_base}/examples/data/breast_hetero_host.csv",
                                          # path="file:///data/projects/fate/examples/data/breast_hetero_host.csv",
                                          format="csv",
                                          match_id_name="id",
                                          delimiter=",",
                                          label_name=None,
                                          dtype="float32")

    deployed_pipeline = pipeline.get_deployed_pipeline()
    deployed_pipeline.intersection_0.input_data = reader_1.outputs["output_data"]

    predict_pipeline.add_task(deployed_pipeline)
    predict_pipeline.add_task(reader_1)

    print("\n\n\n")
    print(predict_pipeline.compile().get_dag())
    predict_pipeline.predict()

    print(predict_pipeline.get_task_info("selection_0").get_output_data())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str, default="",
                        help="config file")
    parser.add_argument("-namespace", type=str, default="",
                        help="namespace for data stored in FATE")
    args = parser.parse_args()
    main(config=args.config, namespace=args.namespace)
