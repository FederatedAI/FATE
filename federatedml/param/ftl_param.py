#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from federatedml.param.base_param import BaseParam


class FTLModelParam(BaseParam):
    """
    Defines parameters for FTL model

    Parameters
    ----------
    max_iteration: integer, default: 10
        The number of passes over the training data (aka epochs), must be positive integer

    eps: numeric, default: 1e-3
        The converge threshold, must be positive number

    alpha: numeric, default: 100
        The weight for objective function loss, must be positive number

    is_encrypt: bool, default; True
        The indicator indicating whether we use encrypted version of ftl or plain version, must be bool

    enc_ftl: str default "dct_enc_ftl"
        The name for encrypted federated transfer learning algorithm

    """

    def __init__(self, max_iteration=10, batch_size=64, eps=1e-5,
                 alpha=100, lr_decay=0.001, l2_para=1, is_encrypt=True, enc_ftl="dct_enc_ftl"):
        self.max_iter = max_iteration
        self.batch_size = batch_size
        self.eps = eps
        self.alpha = alpha
        self.lr_decay = lr_decay
        self.l2_para = l2_para
        self.is_encrypt = is_encrypt
        self.enc_ftl = enc_ftl

    def check(self):
        model_param_descr = "ftl model param's "
        self.check_positive_integer(self.max_iter, model_param_descr + "max_iter")
        self.check_positive_number(self.eps, model_param_descr + "eps")
        self.check_positive_number(self.alpha, model_param_descr + "alpha")
        self.check_boolean(self.is_encrypt, model_param_descr + "is_encrypt")
        return True


class LocalModelParam(BaseParam):
    """
    Defines parameters for FTL model

    Parameters
    ----------
    input_dim: integer, default: None
        The dimension of input samples, must be positive integer

    encode_dim: integer, default: 5
        The dimension of the encoded representation of input samples, must be positive integer

    learning_rate: float, default: 0.001
        The learning rate for training model, must between 0 and 1 exclusively


    """

    def __init__(self, input_dim=None, encode_dim=5, learning_rate=0.001):
        self.input_dim = input_dim
        self.encode_dim = encode_dim
        self.learning_rate = learning_rate

    def check(self):
        model_param_descr = "local model param's "
        if self.input_dim is not None:
            self.check_positive_integer(self.input_dim, model_param_descr + "input_dim")
        self.check_positive_integer(self.encode_dim, model_param_descr + "encode_dim")
        self.check_open_unit_interval(self.learning_rate, model_param_descr + "learning_rate")
        return True


class FTLDataParam(BaseParam):
    """
    Defines parameters for FTL data model

    Parameters
    ----------
    file_path: str, default: None
        The file path to FTL data configuration JSON file, must be string or None

    n_feature_guest: integer, default: 10
        The number of features at guest side, must be positive integer

    n_feature_host: integer, default: 23
        The number of features at host side, must be positive integer

    overlap_ratio: float, default: 0.1
        The ratio of overlapping samples between guest and host, must between 0 and 1 exclusively

    guest_split_ratio: float, default: 0.9
        The ratio of number of samples excluding overlapping samples at guest side, must between 0 and 1 exclusively

    num_samples: numeric, default: None
        The total number of samples used for train/validation/test, must be positive integer or None. If None, all samples
        would be used.

    balanced: bool, default; True
        The indicator indicating whether balance samples, must be bool

    is_read_table: bool, default; False
        The indicator indicating whether read data from dtable, must be bool

    """

    def __init__(self, file_path=None, n_feature_guest=10, n_feature_host=23, overlap_ratio=0.1, guest_split_ratio=0.9,
                 num_samples=None, balanced=True, is_read_table=False):
        self.file_path = file_path
        self.n_feature_guest = n_feature_guest
        self.n_feature_host = n_feature_host
        self.overlap_ratio = overlap_ratio
        self.guest_split_ratio = guest_split_ratio
        self.num_samples = num_samples
        self.balanced = balanced
        self.is_read_table = is_read_table

    def check(self):
        model_param_descr = "ftl data model param's "
        if self.file_path is not None:
            self.check_string(self.file_path, model_param_descr + "file_path")
        if self.num_samples is not None:
            self.check_positive_integer(self.num_samples, model_param_descr + "num_samples")

        self.check_positive_integer(self.n_feature_guest, model_param_descr + "n_feature_guest")
        self.check_positive_integer(self.n_feature_host, model_param_descr + "n_feature_host")
        self.check_boolean(self.balanced, model_param_descr + "balanced")
        self.check_boolean(self.is_read_table, model_param_descr + "is_read_table")
        self.check_open_unit_interval(self.overlap_ratio, model_param_descr + "overlap_ratio")
        self.check_open_unit_interval(self.guest_split_ratio, model_param_descr + "guest_split_ratio")
        return True


class FTLValidDataParam(BaseParam):
    """
    Defines parameters for FTL validation data model

    Parameters
    ----------
    file_path: str, default: None
        The file path to FTL data configuration JSON file, must be string or None

    num_samples: numeric, default: None
        The total number of samples used for validation, must be positive integer or None. If None, all samples
        would be used.

    is_read_table: bool, default; False
        The indicator indicating whether read data from dtable, must be bool

    """

    def __init__(self, file_path=None, num_samples=None, is_read_table=False):
        self.file_path = file_path
        self.num_samples = num_samples
        self.is_read_table = is_read_table

    def check(self):
        model_param_descr = "ftl validation data model param's "
        if self.file_path is not None:
            self.check_string(self.file_path, model_param_descr + "file_path")
        if self.num_samples is not None:
            self.check_positive_integer(self.num_samples, model_param_descr + "num_samples")

        self.check_boolean(self.is_read_table, model_param_descr + "is_read_table")
        return True

