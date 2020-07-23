#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  added by jsweng
#  base class for OHE alignment

import functools

from arch.api.utils import log_utils
from federatedml.feature.one_hot_encoder import OneHotEncoder
from federatedml.param.onehot_encoder_with_alignment_param import OHEAlignmentParam
from federatedml.secureprotol import PaillierEncrypt, FakeEncrypt
from federatedml.transfer_variable.transfer_class.ohe_alignment_transfer_variable import OHEAlignmentTransferVariable
from federatedml.util import consts

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
        self.transfer_variable = OHEAlignmentTransferVariable()

    def init_schema(self, data_instance):
        if data_instance is None:
            return

        self.schema = data_instance.schema
        self.header = self.schema.get('header')

    def _init_params(self, data_instances):
        if data_instances is None:
            return

        super(OHEAlignmentBase, self)._init_params(data_instances)

    def fit(self, data_instances):
        """This function allows for one-hot-encoding of the 
        columns with or without alignment with the other parties
        in the federated learning.

        Args:
            data_instances: data the guest has access to

        Returns:
            if alignment is on, then the one-hot-encoding data_instances are done with
            alignment with parties involved in federated learning else,
            the data is one-hot-encoded independently

        """

        self.init_schema(data_instances)
        self._init_params(data_instances)
        # keep a copy of original header
        ori_header = self.inner_param.header.copy()

        # obtain the individual column headers with their values
        f1 = functools.partial(self.record_new_header,
                               inner_param=self.inner_param)
        self.col_maps = data_instances.mapPartitions(f1).reduce(self.merge_col_maps)
        col_maps = {}
        for col_name, pair_obj in self.col_maps.items():
            values = [str(x) for x in pair_obj.values]
            col_maps[col_name] = values

        LOGGER.debug("new col_maps is: {}".format(col_maps))

        if self.need_alignment:

            # Send col_maps to arbiter
            if self.role == consts.HOST:
                self.transfer_variable.host_columns.remote(col_maps, role=consts.ARBITER, idx=-1)
            elif self.role == consts.GUEST:
                self.transfer_variable.guest_columns.remote(col_maps, role=consts.ARBITER, idx=-1)

                # Receive aligned columns from arbiter
            aligned_columns = self.transfer_variable.aligned_columns.get(idx=-1)
            aligned_col_maps = aligned_columns[0]
            LOGGER.debug("{} aligned columns received are: {}".format(self.role, aligned_col_maps))

            # All the headers - original or new after alignment - are appended together
            new_header = []
            transform_col_names = []
            for col in ori_header:
                if col not in aligned_col_maps:
                    new_header.append(col)
                    continue
                transform_col_names.append(col)
                for vv in aligned_col_maps[col]:
                    new_header.append(col + '_' + vv)

            LOGGER.debug(
                "new transform col names after format received aligned columns: {}".format(transform_col_names))
            LOGGER.debug("new header after format received aligned columns: {}".format(new_header))

            self.inner_param.add_transform_names = transform_col_names
            self.inner_param.set_result_header(new_header)

            LOGGER.debug("Before set_schema in fit, schema is : {}, header: {}".format(self.schema,
                                                                                       self.inner_param.header))

        else:
            self._transform_schema()

        data_instances = self.transform(data_instances)
        LOGGER.debug(
            "[Result][OHEAlignment{}] After transform in fit, schema is : {}, header: {}".format(self.role, self.schema,
                                                                                                 self.inner_param.header))

        return data_instances


class OHEAlignmentGuest(OHEAlignmentBase):
    def __init__(self):
        super(OHEAlignmentGuest, self).__init__()
        self.role = consts.GUEST


class OHEAlignmentHost(OHEAlignmentBase):
    def __init__(self):
        super(OHEAlignmentHost, self).__init__()
        self.role = consts.HOST
