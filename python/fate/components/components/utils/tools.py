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
from .consts import TRAIN_SET, VALIDATE_SET, TEST_SET


TYPE = "type"


def cat_train_and_validate_df(train_df: DataFrame, val_df: DataFrame):
    """
    Concatenate train and validate dataframe
    """
    return train_df.vstack(val_df)


def add_dataset_type(df: DataFrame, dataset_type):
    assert dataset_type in [
        TRAIN_SET,
        VALIDATE_SET,
        TEST_SET,
    ], f"dataset_type must be one of {TRAIN_SET}, {VALIDATE_SET}, {TEST_SET}"
    df[TYPE] = dataset_type
    return df
