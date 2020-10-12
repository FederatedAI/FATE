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

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score

from pipeline.utils.tools import JobConfig


def main(param="./xgb_config_binary.yaml"):
    # obtain config
    if isinstance(param, str):
        param = JobConfig.load_from_file(param)
    data_guest = param["data_guest"]
    data_host = param["data_host"]
    data_test = param["data_test"]

    idx = param["idx"]
    label_name = param["label_name"]

    # prepare data
    df_guest = pd.read_csv(data_guest, index_col=idx)
    df_host = pd.read_csv(data_host, index_col=idx)
    df_test = pd.read_csv(data_test, index_col=idx)

    df = pd.concat([df_guest, df_host], axis=0)
    y = df[label_name]
    X = df.drop(label_name, axis=1)
    y_test = df_test[label_name]
    X_test = df_test.drop(columns=[label_name])

    train_data = xgb.DMatrix(data=X, label=y)
    validate_data = xgb.DMatrix(data=X_test, label=y_test)
    xgb_param = {'max_depth': 3, "eta": 0.1, 'objective': 'binary:logistic'}
    eval_list = [(train_data, 'train')]
    boosting_round = 8

    xgb_model = xgb.train(xgb_param, train_data, num_boost_round=boosting_round, evals=eval_list)
    y_prob = xgb_model.predict(train_data)

    try:
        auc_score = roc_auc_score(y, y_prob)
    except:
        print(f"no auc score available")
        return

    result = {"auc": auc_score}

    return {}, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY SKLEARN JOB")
    parser.add_argument("-param", type=str,
                        help="config file for params")
    args = parser.parse_args()
    if args.config is not None:
        main(args.param)
    main()
