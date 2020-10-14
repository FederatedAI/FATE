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
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


from pipeline.utils.tools import JobConfig


def main(param=""):
    # obtain config
    if isinstance(param, str):
        param = JobConfig.load_from_file(param)
    data_guest = param["data_guest"]
    data_host = param["data_host"]

    idx = param["idx"]
    label_name = param["label_name"]

    # prepare data
    df_guest = pd.read_csv(data_guest, index_col=idx)
    df_host = pd.read_csv(data_host, index_col=idx)
    df = df_guest.join(df_host, rsuffix='host')
    y = df[label_name]
    X = df.drop(label_name, axis=1)
    clf = GradientBoostingRegressor(random_state=0, n_estimators=50)
    clf.fit(X, y)

    y_predict = clf.predict(X)

    result = {"mean_squared_error": mean_squared_error(y, y_predict),
              "mean_absolute_error": mean_absolute_error(y, y_predict),
              }
    print(result)
    return {}, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY SKLEARN JOB")
    parser.add_argument("-param", type=str,
                        help="config file for params")
    args = parser.parse_args()
    if args.config is not None:
        main(args.param)
    main()
