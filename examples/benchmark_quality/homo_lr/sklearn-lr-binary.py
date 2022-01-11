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

import pandas
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, roc_curve

from pipeline.utils.tools import JobConfig
import os


def main(config="../../config.yaml", param="./lr_config.yaml"):
    # obtain config
    if isinstance(param, str):
        param = JobConfig.load_from_file(param)
    assert isinstance(param, dict)
    data_guest = param["data_guest"]
    data_host = param["data_host"]
    data_test = param["data_test"]
    idx = param["idx"]
    label_name = param["label_name"]

    if isinstance(config, str):
        config = JobConfig.load_from_file(config)
        data_base_dir = config["data_base_dir"]
    else:
        data_base_dir = config.data_base_dir

    config_param = {
        "penalty": param["penalty"],
        "max_iter": 100,
        "alpha": param["alpha"],
        "learning_rate": "optimal",
        "eta0": param["learning_rate"]
    }

    # prepare data
    df_guest = pandas.read_csv(os.path.join(data_base_dir, data_guest), index_col=idx)
    df_host = pandas.read_csv(os.path.join(data_base_dir, data_host), index_col=idx)

    # df_test = pandas.read_csv(data_test, index_col=idx)

    df = pandas.concat([df_guest, df_host], axis=0)

    # df = df_guest.join(df_host, rsuffix="host")
    y_train = df[label_name]
    x_train = df.drop(label_name, axis=1)

    # y_test = df_test[label_name]
    # x_test = df_test.drop(label_name, axis=1)
    x_test, y_test = x_train, y_train

    # lm = LogisticRegression(max_iter=20)
    lm = SGDClassifier(loss="log", **config_param)
    lm_fit = lm.fit(x_train, y_train)
    y_pred = lm_fit.predict(x_test)
    y_prob = lm_fit.predict_proba(x_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred, average="macro")
    pr = precision_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    # y_predict_proba = est.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    ks = max(tpr - fpr)
    result = {"auc": auc_score}
    print(f"result: {result}")
    print(f"coef_: {lm_fit.coef_}, intercept_: {lm_fit.intercept_}, n_iter: {lm_fit.n_iter_}")

    return {}, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY SKLEARN JOB")
    parser.add_argument("-p", "--param", type=str, default="./lr_config.yaml",
                        help="config file for params")
    args = parser.parse_args()
    main(args.param)
