#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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


import logging

from fate.arch import Context
from fate.arch.dataframe import DataFrame
from ..abc.module import Module

logger = logging.getLogger(__name__)


class DataSplitModuleGuest(Module):
    def __init__(
            self,
            train_size=0.8,
            validate_size=0.2,
            test_size=0.0,
            stratified=False,
            random_state=None,
            hetero_sync=True
    ):
        self.train_size = train_size
        self.validate_size = validate_size
        self.test_size = test_size
        self.stratified = stratified
        self.random_state = random_state
        self.hetero_sync = hetero_sync

    def fit(self, ctx: Context, train_data, validate_data=None):
        data_count = train_data.shape[0]
        train_size, validate_size, test_size = get_split_data_size(self.train_size,
                                                                   self.validate_size,
                                                                   self.test_size,
                                                                   data_count)
        if self.stratified:
            train_data_set = sample_per_label(train_data, sample_count=train_size, random_state=self.random_state)
        else:
            train_data_set = sample_data(df=train_data, n=train_size, random_state=self.random_state)
        if train_data_set is not None:
            train_sid = train_data_set.get_indexer(target="sample_id")
            validate_test_data_set = train_data.drop(train_data_set)
        else:
            train_sid = None
            validate_test_data_set = train_data

        if self.stratified:
            validate_data_set = sample_per_label(validate_test_data_set, sample_count=validate_size,
                                                 random_state=self.random_state)
        else:
            validate_data_set = sample_data(df=validate_test_data_set, n=validate_size, random_state=self.random_state)
        if validate_data_set is not None:
            validate_sid = validate_data_set.get_indexer(target="sample_id")
            test_data_set = validate_test_data_set.drop(validate_data_set)
            if test_data_set.shape[0] == 0:
                test_sid = None
                test_data_set = None
            else:
                test_sid = test_data_set.get_indexer(target="sample_id")
        else:
            validate_sid = None
            if validate_test_data_set.shape[0] == 0:
                test_data_set = None
                test_sid = None
            else:
                test_data_set = validate_test_data_set
                test_sid = validate_test_data_set.get_indexer(target="sample_id")

        if self.hetero_sync:
            ctx.hosts.put("train_data_sid", train_sid)
            ctx.hosts.put("validate_data_sid", validate_sid)
            ctx.hosts.put("test_data_sid", test_sid)

        return train_data_set, validate_data_set, test_data_set


class DataSplitModuleHost(Module):
    def __init__(
            self,
            train_size=0.8,
            validate_size=0.2,
            test_size=0.0,
            stratified=False,
            random_state=None,
            hetero_sync=True
    ):
        self.train_size = train_size
        self.validate_size = validate_size
        self.test_size = test_size
        self.stratified = stratified
        self.random_state = random_state
        self.hetero_sync = hetero_sync

    def fit(self, ctx: Context, train_data, validate_data=None):
        if self.hetero_sync:
            train_data_sid = ctx.guest.get("train_data_sid")
            validate_data_sid = ctx.guest.get("validate_data_sid")
            test_data_sid = ctx.guest.get("test_data_sid")
            train_data_set, validate_data_set, test_data_set = None, None, None
            if train_data_sid:
                train_data_set = train_data.loc(train_data_sid, preserve_order=True)
            if validate_data_sid:
                validate_data_set = train_data.loc(validate_data_sid, preserve_order=True)
            if test_data_sid:
                test_data_set = train_data.loc(test_data_sid, preserve_order=True)
        else:
            data_count = train_data.shape[0]
            train_size, validate_size, test_size = get_split_data_size(self.train_size,
                                                                       self.validate_size,
                                                                       self.test_size,
                                                                       data_count)

            if self.stratified:
                train_data_set = sample_per_label(train_data, sample_count=train_size, random_state=self.random_state)
            else:
                train_data_set = sample_data(df=train_data, n=train_size, random_state=self.random_state)
            if train_data_set is not None:
                # train_sid = train_data_set.get_indexer(target="sample_id")
                validate_test_data_set = train_data.drop(train_data_set)
            else:
                validate_test_data_set = train_data

            if self.stratified:
                validate_data_set = sample_per_label(validate_test_data_set, sample_count=validate_size,
                                                     random_state=self.random_state)
            else:
                validate_data_set = sample_data(df=validate_test_data_set, n=validate_size,
                                                random_state=self.random_state)
            if validate_data_set is not None:
                # validate_sid = validate_data_set.get_indexer(target="sample_id")
                test_data_set = validate_test_data_set.drop(validate_data_set)
                if test_data_set.shape[0] == 0:
                    test_data_set = None
            else:
                if validate_test_data_set.shape[0] == 0:
                    test_data_set = None
                else:
                    test_data_set = validate_test_data_set

        return train_data_set, validate_data_set, test_data_set


def sample_data(df, n, random_state):
    if n == 0:
        return
    else:
        return df.sample(n=n, random_state=random_state)


def sample_per_label(train_data, sample_count=None, random_state=None):
    train_data_binarized_label = train_data.label.get_dummies()
    labels = [label_name.split("_")[1] for label_name in train_data_binarized_label.columns]
    sampled_data_df = []
    sampled_n = 0
    data_n = train_data.shape[0]
    for i, label in enumerate(labels):
        label_data = train_data.iloc(train_data.label == int(label))
        if i == len(labels) - 1:
            # last label:
            to_sample_n = sample_count - sampled_n
        else:
            to_sample_n = round(label_data.shape[0] / data_n * sample_count)
        label_sampled_data = sample_data(df=label_data, n=to_sample_n, random_state=random_state)
        if label_sampled_data is not None:
            sampled_data_df.append(label_sampled_data)
            sampled_n += label_sampled_data.shape[0]
    sampled_data = None
    if sampled_data_df:
        sampled_data = DataFrame.vstack(sampled_data_df)
    return sampled_data


def get_split_data_size(train_size, validate_size, test_size, data_count):
    """
    Validate & transform param inputs into all int
    """
    # check & transform data set sizes
    if isinstance(test_size, float) or isinstance(train_size, float) or isinstance(validate_size, float):
        total_size = 1.0
    else:
        total_size = data_count
    if train_size is None:
        if validate_size is None:
            train_size = total_size - test_size
            validate_size = total_size - (test_size + train_size)
        else:
            if test_size is None:
                test_size = 0
            train_size = total_size - (validate_size + test_size)
    elif test_size is None:
        if validate_size is None:
            test_size = total_size - train_size
            validate_size = total_size - (test_size + train_size)
        else:
            test_size = total_size - (validate_size + train_size)
    elif validate_size is None:
        if train_size is None:
            train_size = total_size - test_size
        validate_size = total_size - (test_size + train_size)

    if abs((abs(train_size) + abs(test_size) + abs(validate_size)) - total_size) > 1e-6:
        raise ValueError(f"train_size, test_size, validate_size should sum up to 1.0 or data count")

    if isinstance(train_size, float):
        train_size = round(train_size * data_count)
        validate_size = round(validate_size * data_count)
        test_size = total_size - train_size - validate_size
    return train_size, validate_size, test_size
