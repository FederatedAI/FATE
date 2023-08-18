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

import numpy as np
import pandas
from fate_client.pipeline.utils.test_utils import JobConfig
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


def main(config="../../config.yaml", param="./linr_sklearn_config.yaml"):
    # obtain config
    if isinstance(param, str):
        param = JobConfig.load_from_file(param)
    data_guest = param["data_guest"]
    data_host = param["data_host"]
    idx = param["idx"]
    label_name = param["label_name"]

    if isinstance(config, str):
        config = JobConfig.load_from_file(config)
        print(f"config: {config}")
        data_base_dir = config["data_base_dir"]
    else:
        data_base_dir = config.data_base_dir

    # prepare data
    df_guest = pandas.read_csv(os.path.join(data_base_dir, data_guest), index_col=idx)
    df_host = pandas.read_csv(os.path.join(data_base_dir, data_host), index_col=idx)
    df = df_guest.join(df_host, rsuffix="host")
    y = df[label_name]
    X = df.drop(label_name, axis=1)
    lm = SGDRegressor(loss="squared_error", penalty=param["penalty"], random_state=42,
                      fit_intercept=True, max_iter=param["epochs"], average=param["batch_size"])
    lm_fit = lm.fit(X, y)
    y_pred = lm_fit.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    explained_var = explained_variance_score(y, y_pred)
    metric_summary = {"r2_score": r2,
                      "mse": mse,
                      "rmse": rmse}
    data_summary = {}
    return data_summary, metric_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY LOCAL JOB")
    parser.add_argument("-c", "--config", type=str,
                        help="config file", default="../../config.yaml")
    parser.add_argument("-p", "--param", type=str,
                        help="config file for params", default="./linr_sklearn_config.yaml")
    args = parser.parse_args()
    main(args.config, args.param)
