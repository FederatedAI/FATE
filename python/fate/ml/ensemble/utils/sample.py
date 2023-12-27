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

from fate.arch.dataframe import DataFrame


def goss_sample(gh: DataFrame, top_rate: float, other_rate: float, random_seed=42):
    # check param, top rate + other rate <= 1, and they must be float
    assert isinstance(top_rate, float), "top rate must be float, but got {}".format(type(top_rate))
    assert isinstance(other_rate, float), "other rate must be float, but got {}".format(type(other_rate))
    assert top_rate + other_rate <= 1, "top rate + other rate must <= 1, but got {}".format(top_rate + other_rate)
    sample_num = len(gh)
    a_part_num = int(sample_num * top_rate)
    b_part_num = int(sample_num * other_rate)
    top_samples = gh.nlargest(n=a_part_num, columns=["g"], error=0)
    rest_samples = gh.drop(top_samples).sample(n=b_part_num, random_state=random_seed)
    sampled_rs = DataFrame.vstack([top_samples, rest_samples])
    return sampled_rs
