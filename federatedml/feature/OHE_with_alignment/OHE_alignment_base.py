#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  added by jsweng
#  base class for OHE alignment

import functools

from arch.api.utils import log_utils
from federatedml.feature.one_hot_encoder import OneHotEncoder
from federatedml.feature.one_hot_encoder import OneHotInnerParam
from federatedml.param.onehot_encoder_with_alignment_param import OHEAlignmentParam
from federatedml.transfer_variable.transfer_class.OHE_alignment_transfer_variable import OHEAlignmentTransferVariable
from federatedml.util import consts
from federatedml.util import fate_operator
from federatedml.protobuf.generated import onehot_param_pb2, onehot_meta_pb2
from federatedml.statistic.data_overview import get_header

from federatedml.secureprotol import PaillierEncrypt, FakeEncrypt
from federatedml.statistic import data_overview

LOGGER = log_utils.getLogger()


class OHEAlignmentBase(OneHotEncoder):
    def __init__(self):
        super(OHEAlignmentBase, self).__init__()
        self.model_name = 'OHEAlignment'
        self.model_param_name = 'OHEAlignmentParam'
        self.model_meta_name = 'OHEAlignmentMeta'
        self.model_param = OHEAlignmentParam()


    def _init_model(self, params):
        super(OHEAlignmentBase, self)._init_model(params)
        # self.re_encrypt_batches = params.re_encrypt_batches
        self.need_alignment = params.need_alignment

        if params.encrypt_param.method == consts.PAILLIER:
            self.cipher_operator = PaillierEncrypt()
        else:
            self.cipher_operator = FakeEncrypt()

        self.transfer_variable = OHEAlignmentTransferVariable()
    
    def init_schema(self, data_instance):
        if data_instance is None:
            return
        self.schema = data_instance.schema
        self.header = self.schema.get('header')

    def _init_params(self, data_instances):
        if data_instances is None: 
            return 

        if len(self.schema) == 0:
            self.schema = data_instances.schema

        if self.inner_param is not None:
            return
        self.inner_param = OneHotInnerParam()
        # self.schema = data_instances.schema
        LOGGER.debug("In _init_params, schema is : {}".format(self.schema))
        header = get_header(data_instances)
        self.inner_param.set_header(header)

        if self.model_param.transform_col_indexes == -1:
            self.inner_param.set_transform_all()
        else:
            self.inner_param.add_transform_indexes(self.model_param.transform_col_indexes)
            self.inner_param.add_transform_names(self.model_param.transform_col_names)
    
    def _transform_schema(self):
        header = self.inner_param.header.copy()
        LOGGER.debug("**AIFEL** [Result][OneHotEncoder]Before one-hot, "
                     "data_instances schema is : {}".format(self.inner_param.header))
        result_header = []
        for col_name in header:
            if col_name not in self.col_maps:
                result_header.append(col_name)
                continue
            pair_obj = self.col_maps[col_name]

            new_headers = pair_obj.transformed_headers
            result_header.extend(new_headers)

        self.inner_param.set_result_header(result_header)
        LOGGER.debug("**AIFEL** [Result][OneHotEncoder]After one-hot, data_instances schema is : {}".format(result_header))

    