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
################################################################################
#
#
################################################################################

# =============================================================================
# Sparse Feature
# =============================================================================


class SparseVector(object):
    """
    Sparse storage data format of federatedml

    Parameters
    ----------
    sparse_vec : dict, record (indice, data) kv tuples

    shape : the real feature shape of data

    """
    def __init__(self, indices=None, data=None, shape=0):
        self.sparse_vec = dict(zip(indices, data))
        self.shape = shape

    def get_data(self, pos, default_val=None):
        return self.sparse_vec.get(pos, default_val)

    def count_non_zeros(self):
        return len(self.sparse_vec)

    def count_zeros(self):
        return self.shape - len(self.sparse_vec)

    def get_shape(self):
        return self.shape

    def get_all_data(self):
        for idx, data in self.sparse_vec.items():
            yield idx, data

    def get_sparse_vector(self):
        return self.sparse_vec

    def set_sparse_vector(self, sparse_vec):
        self.sparse_vec = sparse_vec
