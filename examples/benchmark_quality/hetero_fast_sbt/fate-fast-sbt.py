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
from pipeline.component.hetero_fast_secureboost import HeteroFastSecureBoost
from pipeline.component.intersection import Intersection
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.component.evaluation import Evaluation
from pipeline.interface.model import Model
from pipeline.utils.tools import load_job_config
from pipeline.utils.tools import JobConfig

from federatedml.evaluation.metrics import regression_metric, classification_metric
from fate_test.utils import extract_data, parse_summary_result


def main(config="../../config.yaml", param="./xgb_config_binary.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)

    if isinstance(param, str):
        param = JobConfig.load_from_file(param)

    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]

    # data sets
    guest_train_data = {"name": param['data_guest_train'], "namespace": f"experiment{namespace}"}
    host_train_data = {"name": param['data_host_train'], "namespace": f"experiment{namespace}"}
    guest_validate_data = {"name": param['data_guest_val'], "namespace": f"experiment{namespace}"}
    host_validate_data = {"name": param['data_host_val'], "namespace": f"experiment{namespace}"}

    # init pipeline
    pipeline = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest, host=host,)

    # set data reader and data-io

    reader_0, reader_1 = Reader(name="reader_0"), Reader(name="reader_1")
    reader_0.get_party_instance(role="guest", party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role="host", party_id=host).component_param(table=host_train_data)
    reader_1.get_party_instance(role="guest", party_id=guest).component_param(table=guest_validate_data)
    reader_1.get_party_instance(role="host", party_id=host).component_param(table=host_validate_data)

    dataio_0, dataio_1 = DataIO(name="dataio_0"), DataIO(name="dataio_1")

    dataio_0.get_party_instance(role="guest", party_id=guest).component_param(with_label=True, output_format="dense")
    dataio_0.get_party_instance(role="host", party_id=host).component_param(with_label=False)
    dataio_1.get_party_instance(role="guest", party_id=guest).component_param(with_label=True, output_format="dense")
    dataio_1.get_party_instance(role="host", party_id=host).component_param(with_label=False)

    # data intersect component
    intersect_0 = Intersection(name="intersection_0")
    intersect_1 = Intersection(name="intersection_1")

    # secure boost component
    hetero_fast_sbt_0 = HeteroFastSecureBoost(name="hetero_fast_sbt_0",
                                              num_trees=param['tree_num'],
                                              task_type=param['task_type'],
                                              objective_param={"objective": param['loss_func']},
                                              encrypt_param={"method": "Paillier"},
                                              tree_param={"max_depth": param['tree_depth']},
                                              validation_freqs=1,
                                              subsample_feature_rate=1,
                                              learning_rate=param['learning_rate'],
                                              guest_depth=param['guest_depth'],
                                              host_depth=param['host_depth'],
                                              tree_num_per_party=param['tree_num_per_party'],
                                              work_mode=param['work_mode']
                                              )
    hetero_fast_sbt_1 = HeteroFastSecureBoost(name="hetero_fast_sbt_1")
    # evaluation component
    evaluation_0 = Evaluation(name="evaluation_0", eval_type=param['eval_type'])

    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(dataio_1, data=Data(data=reader_1.output.data), model=Model(dataio_0.output.model))
    pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(intersect_1, data=Data(data=dataio_1.output.data))
    pipeline.add_component(hetero_fast_sbt_0, data=Data(train_data=intersect_0.output.data,
                                                        validate_data=intersect_1.output.data))
    pipeline.add_component(hetero_fast_sbt_1, data=Data(test_data=intersect_1.output.data),
                           model=Model(hetero_fast_sbt_0.output.model))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_fast_sbt_0.output.data))

    pipeline.compile()
    pipeline.fit()

    sbt_0_data = pipeline.get_component("hetero_fast_sbt_0").get_output_data().get("data")
    sbt_1_data = pipeline.get_component("hetero_fast_sbt_1").get_output_data().get("data")
    sbt_0_score = extract_data(sbt_0_data, "predict_result")
    sbt_0_label = extract_data(sbt_0_data, "label")
    sbt_1_score = extract_data(sbt_1_data, "predict_result")
    sbt_1_label = extract_data(sbt_1_data, "label")
    sbt_0_score_label = extract_data(sbt_0_data, "predict_result", keep_id=True)
    sbt_1_score_label = extract_data(sbt_1_data, "predict_result", keep_id=True)
    metric_summary = parse_summary_result(pipeline.get_component("evaluation_0").get_summary())
    if param['eval_type'] == "regression":
        desc_sbt_0 = regression_metric.Describe().compute(sbt_0_score)
        desc_sbt_1 = regression_metric.Describe().compute(sbt_1_score)
        metric_summary["script_metrics"] = {"hetero_fast_sbt_train": desc_sbt_0,
                                            "hetero_fast_sbt_validate": desc_sbt_1}
    elif param['eval_type'] == "binary":
        metric_sbt = {
            "score_diversity_ratio": classification_metric.Distribution.compute(sbt_0_score_label, sbt_1_score_label),
            "ks_2samp": classification_metric.KSTest.compute(sbt_0_score, sbt_1_score),
            "mAP_D_value": classification_metric.AveragePrecisionScore().compute(sbt_0_score, sbt_1_score, sbt_0_label,
                                                                                 sbt_1_label)}
        metric_summary["distribution_metrics"] = {"hetero_fast_sbt": metric_sbt}
    elif param['eval_type'] == "multi":
        metric_sbt = {
            "score_diversity_ratio": classification_metric.Distribution.compute(sbt_0_score_label, sbt_1_score_label)}
        metric_summary["distribution_metrics"] = {"hetero_fast_sbt": metric_sbt}

    data_summary = {"train": {"guest": guest_train_data["name"], "host": host_train_data["name"]},
                    "test": {"guest": guest_train_data["name"], "host": host_train_data["name"]}
                    }

    return data_summary, metric_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY PIPELINE JOB")
    parser.add_argument("-config", type=str,
                        help="config file")
    parser.add_argument("-param", type=str,
                        help="config file for params")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config, args.param)
    else:
        main()
