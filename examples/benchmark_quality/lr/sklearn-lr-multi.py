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
import os

import pandas
from pipeline.utils.tools import JobConfig
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score


def main(config="../../config.yaml", param="./vechile_config.yaml"):
    # obtain config
    if isinstance(param, str):
        param = JobConfig.load_from_file(param)
    assert isinstance(param, dict)
    data_guest = param["data_guest"]
    data_host = param["data_host"]

    idx = param["idx"]
    label_name = param["label_name"]

    if isinstance(config, str):
        config = JobConfig.load_from_file(config)
        data_base_dir = config["data_base_dir"]
    else:
        data_base_dir = config.data_base_dir

    config_param = {
        "penalty": param["penalty"],
        "max_iter": param["max_iter"],
        "alpha": param["alpha"],
        "learning_rate": "optimal",
        "eta0": param["learning_rate"],
        "random_state": 105
    }

    # prepare data
    df_guest = pandas.read_csv(os.path.join(data_base_dir, data_guest), index_col=idx)
    df_host = pandas.read_csv(os.path.join(data_base_dir, data_host), index_col=idx)

    df = df_guest.join(df_host, rsuffix="host")
    y = df[label_name]
    X = df.drop(label_name, axis=1)
    # lm = LogisticRegression(max_iter=20)
    lm = SGDClassifier(loss="log", **config_param, shuffle=False)
    lm_fit = lm.fit(X, y)
    y_pred = lm_fit.predict(X)

    recall = recall_score(y, y_pred, average="macro")
    pr = precision_score(y, y_pred, average="macro")
    acc = accuracy_score(y, y_pred)

    result = {"accuracy": acc}
    print(result)
    return {}, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY SKLEARN JOB")
    parser.add_argument("-param", type=str,
                        help="config file for params")
    args = parser.parse_args()
    if args.param is not None:
        main(args.param)
