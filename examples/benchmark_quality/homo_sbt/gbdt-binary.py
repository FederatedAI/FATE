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
import os
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from pipeline.utils.tools import JobConfig


def main(config="../../config.yaml", param="./gbdt_config_binary.yaml"):
    # obtain config
    if isinstance(param, str):
        param = JobConfig.load_from_file(param)

    data_guest = param["data_guest"]
    data_host = param["data_host"]
    idx = param["idx"]
    label_name = param["label_name"]

    print('config is {}'.format(config))
    if isinstance(config, str):
        config = JobConfig.load_from_file(config)
        data_base_dir = config["data_base_dir"]
        print('data base dir is', data_base_dir)
    else:
        data_base_dir = config.data_base_dir

    # prepare data
    df_guest = pd.read_csv(os.path.join(data_base_dir, data_guest), index_col=idx)
    df_host = pd.read_csv(os.path.join(data_base_dir, data_host), index_col=idx)
    df = pd.concat([df_guest, df_host], axis=0)
    y = df[label_name]
    X = df.drop(label_name, axis=1)
    X_guest = df_guest.drop(label_name, axis=1)
    y_guest = df_guest[label_name]
    clf = GradientBoostingClassifier(n_estimators=120 if 'epsilon' in data_guest else 50, learning_rate=0.1)
    clf.fit(X, y)
    y_prob = clf.predict(X_guest)

    try:
        auc_score = roc_auc_score(y_guest, y_prob)
    except BaseException:
        print(f"no auc score available")
        return

    result = {"auc": auc_score}
    import time
    print(result)
    print(data_guest)
    time.sleep(3)
    return {}, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY SKLEARN JOB")
    parser.add_argument("-param", type=str,
                        help="config file for params")
    args = parser.parse_args()
    if args.config is not None:
        main(args.param)
    main()
