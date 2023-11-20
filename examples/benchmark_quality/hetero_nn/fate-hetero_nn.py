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
import torch as t
from torch import nn
from pipeline import fate_torch_hook
from pipeline.backend.pipeline import PipeLine
from pipeline.component.nn import DatasetParam
from pipeline.component import DataTransform
from pipeline.component import Evaluation
from pipeline.component import HeteroNN
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data, Model
from pipeline.utils.tools import load_job_config, JobConfig
from federatedml.evaluation.metrics import classification_metric
from fate_test.utils import extract_data, parse_summary_result

fate_torch_hook(t)


def build(param, shape1, shape2):

    guest_bottom = t.nn.Sequential(
        nn.Linear(shape1, param["bottom_layer_units"]),
        nn.ReLU()
    )

    host_bottom = t.nn.Sequential(
        nn.Linear(shape2, param["bottom_layer_units"]),
        nn.ReLU()
    )

    interactive_layer = t.nn.InteractiveLayer(
        guest_dim=param["bottom_layer_units"],
        host_dim=param["bottom_layer_units"],
        host_num=1,
        out_dim=param["interactive_layer_units"])

    act = nn.Sigmoid() if param["top_layer_units"] == 1 else nn.Softmax(dim=1)
    top_layer = t.nn.Sequential(
        t.nn.Linear(
            param["interactive_layer_units"],
            param["top_layer_units"]),
        act)

    return guest_bottom, host_bottom, interactive_layer, top_layer


def main(
        config="../../config.yaml",
        param="./hetero_nn_breast_config.yaml",
        namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)

    if isinstance(param, str):
        param = JobConfig.load_from_file(param)

    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]

    guest_train_data = {
        "name": param["guest_table_name"],
        "namespace": f"experiment{namespace}"}
    host_train_data = {
        "name": param["host_table_name"],
        "namespace": f"experiment{namespace}"}

    pipeline = PipeLine().set_initiator(
        role='guest', party_id=guest).set_roles(
        guest=guest, host=host)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(
        role='guest',
        party_id=guest).component_param(
        table=guest_train_data)
    reader_0.get_party_instance(
        role='host',
        party_id=host).component_param(
        table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(
        role='guest',
        party_id=guest).component_param(
        with_label=True)

    data_transform_0.get_party_instance(
        role='host', party_id=host).component_param(
        with_label=False)

    intersection_0 = Intersection(
        name="intersection_0",
        intersect_method="rsa",
        rsa_params={"hash_method": "sha256", "final_hash_method": "sha256", "key_length": 2048})

    guest_bottom, host_bottom, interactive_layer, top_layer = build(
        param, param['shape1'], param['shape2'])

    if param["loss"] == "categorical_crossentropy":
        loss = t.nn.CrossEntropyLoss()
        ds_param = DatasetParam(
            dataset_name='table',
            flatten_label=True,
            label_dtype='long')
    else:
        loss = t.nn.BCELoss()
        ds_param = DatasetParam(dataset_name='table')

    hetero_nn_0 = HeteroNN(
        name="hetero_nn_0",
        epochs=param["epochs"],
        interactive_layer_lr=param["learning_rate"],
        batch_size=param["batch_size"],
        seed=100,
        encrypt_param={"key_length": 1024},
        dataset=ds_param)
    guest_nn_0 = hetero_nn_0.get_party_instance(role='guest', party_id=guest)
    host_nn_0 = hetero_nn_0.get_party_instance(role='host', party_id=host)

    guest_nn_0.add_bottom_model(guest_bottom)
    guest_nn_0.add_top_model(top_layer)
    host_nn_0.add_bottom_model(host_bottom)
    hetero_nn_0.set_interactive_layer(interactive_layer)
    hetero_nn_0.compile(
        optimizer=t.optim.Adam(
            lr=param['learning_rate']),
        loss=loss)
    hetero_nn_1 = HeteroNN(name="hetero_nn_1")

    if param["loss"] == "categorical_crossentropy":
        eval_type = "multi"
    else:
        eval_type = "binary"

    evaluation_0 = Evaluation(name="evaluation_0", eval_type=eval_type)

    pipeline.add_component(reader_0)
    pipeline.add_component(
        data_transform_0, data=Data(
            data=reader_0.output.data))
    pipeline.add_component(
        intersection_0, data=Data(
            data=data_transform_0.output.data))
    pipeline.add_component(
        hetero_nn_0, data=Data(
            train_data=intersection_0.output.data))
    pipeline.add_component(
        hetero_nn_1, data=Data(
            test_data=intersection_0.output.data), model=Model(
            hetero_nn_0.output.model))
    pipeline.add_component(
        evaluation_0, data=Data(
            data=hetero_nn_0.output.data))

    pipeline.compile()

    pipeline.fit()

    nn_0_data = pipeline.get_component("hetero_nn_0").get_output_data()
    nn_1_data = pipeline.get_component("hetero_nn_1").get_output_data()
    nn_0_score = extract_data(nn_0_data, "predict_result")
    nn_0_label = extract_data(nn_0_data, "label")
    nn_1_score = extract_data(nn_1_data, "predict_result")
    nn_1_label = extract_data(nn_1_data, "label")
    nn_0_score_label = extract_data(nn_0_data, "predict_result", keep_id=True)
    nn_1_score_label = extract_data(nn_1_data, "predict_result", keep_id=True)
    metric_summary = parse_summary_result(
        pipeline.get_component("evaluation_0").get_summary())
    if eval_type == "binary":
        metric_nn = {
            "score_diversity_ratio": classification_metric.Distribution.compute(
                nn_0_score_label,
                nn_1_score_label),
            "ks_2samp": classification_metric.KSTest.compute(
                nn_0_score,
                nn_1_score),
            "mAP_D_value": classification_metric.AveragePrecisionScore().compute(
                nn_0_score,
                nn_1_score,
                nn_0_label,
                nn_1_label)}
        metric_summary["distribution_metrics"] = {"hetero_nn": metric_nn}
    elif eval_type == "multi":
        metric_nn = {
            "score_diversity_ratio": classification_metric.Distribution.compute(
                nn_0_score_label, nn_1_score_label)}
        metric_summary["distribution_metrics"] = {"hetero_nn": metric_nn}

    data_summary = {
        "train": {
            "guest": guest_train_data["name"],
            "host": host_train_data["name"]},
        "test": {
            "guest": guest_train_data["name"],
            "host": host_train_data["name"]}}
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
