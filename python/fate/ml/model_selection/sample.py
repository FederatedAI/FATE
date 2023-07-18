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


class SampleModuleGuest(Module):
    def __init__(
            self,
            mode="random",
            replace=False,
            frac=1.0,
            n=None,
            random_state=None,
            ctx_mode="hetero"
    ):
        self.mode = mode
        self.replace = replace
        self.frac = frac
        self.n = n
        self.random_state = random_state
        self.ctx_mode = ctx_mode

        self._sample_obj = None

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        if self.mode == "stratified":
            # labels = train_data.label.unique()
            train_data_binarized_label = train_data.label.get_dummies()
            labels = [label_name.split("_")[1] for label_name in train_data_binarized_label.columns]
            if self.n is not None:
                label_count_dict = self.n
                if not isinstance(self.n, dict):
                    label_count_dict = {label: self.n for label in labels}
                sampled_data = sample_per_label(train_data, label_count_dict=label_count_dict)
            else:
                label_frac_dict = self.frac
                if not isinstance(self.frac, dict):
                    label_frac_dict = {label: self.frac for label in labels}
                sampled_data = sample_per_label(train_data, label_frac_dict=label_frac_dict)
        elif self.mode == "random":
            if self.n is not None:
                sampled_data = train_data.sample(n=self.n, replace=self.replace, random_state=self.random_state)
            else:
                sampled_data = train_data.sample(frac=self.frac, replace=self.replace, random_state=self.random_state)
        elif self.mode == "weight":
            if self.n is not None:
                sampled_data = train_data.sample(n=self.n,
                                                 replace=self.replace,
                                                 weight=train_data.weight,
                                                 random_state=self.random_state)
            else:
                sampled_data = train_data.sample(frac=self.frac,
                                                 replace=self.replace,
                                                 weight=train_data.weight,
                                                 random_state=self.random_state)

        else:
            raise ValueError(f"Unknown sample mode: {self.mode}")

        if self.ctx_mode == "hetero":
            sampled_mid = sampled_data.match_id
            ctx.hosts.put("sampled_mid", sampled_mid)

        return sampled_data


class SampleModuleHost(Module):
    def __init__(
            self,
            mode="random",
            replace=False,
            frac=1.0,
            n=None,
            random_state=None,
            ctx_mode="hetero"
    ):
        self.mode = mode
        self.replace = replace
        self.frac = frac
        self.n = n
        self.random_state = random_state
        self.ctx_mode = ctx_mode

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        if self.ctx_mode == "hetero":
            sampled_mid = ctx.guest.get("sampled_mid")
            # maybe other api?
            sampled_data = train_data.join(sampled_mid, how="inner")
        elif self.ctx_mode in ["homo", "local"]:
            if self.mode == "stratified":
                labels = train_data.label.unique()
                if self.n is not None:
                    label_count_dict = self.n
                    if not isinstance(self.n, dict):
                        label_count_dict = {label: self.n for label in labels}
                    sampled_data = sample_per_label(train_data, label_count_dict=label_count_dict)
                else:
                    label_frac_dict = self.frac
                    if not isinstance(self.frac, dict):
                        label_frac_dict = {label: self.frac for label in labels}
                    sampled_data = sample_per_label(train_data, label_frac_dict=label_frac_dict)
            elif self.mode == "random":
                if self.n is not None:
                    sampled_data = train_data.sample(n=self.n, replace=self.replace, random_state=self.random_state)
                else:
                    sampled_data = train_data.sample(frac=self.frac,
                                                     replace=self.replace,
                                                     random_state=self.random_state)
            elif self.mode == "weight":
                if self.n is not None:
                    sampled_data = train_data.sample(n=self.n,
                                                     replace=self.replace,
                                                     weight=train_data.weight,
                                                     random_state=self.random_state)
                else:
                    sampled_data = train_data.sample(frac=self.frac,
                                                     relace=self.replace,
                                                     weight=train_data.weight,
                                                     random_state=self.random_state)
            else:
                raise ValueError(f"Unknown sample mode: {self.mode}")
        else:
            raise ValueError(f"Unknown ctx mode: {self.ctx_mode}")

        return sampled_data


def sample_per_label(train_data, label_frac_dict=None, label_count_dict=None, replace=False, random_state=None):
    sampled_data_df = []
    if label_frac_dict is not None:
        labels = label_frac_dict.keys()
        for label in labels:
            label_frac = label_frac_dict[label]
            label_data = train_data[train_data[train_data.schema.label_name] == label]
            label_sampled_data = label_data.sample(frac=label_frac, replace=replace, random_state=random_state)
            sampled_data_df.append(label_sampled_data)
    elif label_count_dict is not None:
        labels = label_count_dict.keys()
        for label in labels:
            label_count = label_count_dict[label]
            label_data = train_data[train_data[train_data.schema.label_name] == label]
            label_sampled_data = label_data.sample(n=label_count, reaplce=replace, random_state=random_state)
            sampled_data_df.append(label_sampled_data)
    else:
        raise ValueError("label_frac_dict and label_count_dict can not be None at the same time")
    sampled_data = DataFrame.vstack(sampled_data_df)
    return sampled_data
