#!/usr/bin/env python
# -*- coding: utf-8 -*-


#
#  added by jsweng
#  alignemnet arbiter

import numpy as np

from arch.api.utils import log_utils
from federatedml.framework.homo.procedure import aggregator
from federatedml.framework.homo.procedure import paillier_cipher
from federatedml.feature.OHE_with_alignment.OHE_alignment_base import OHEAlignmentBase
from federatedml.util import consts
import ast
from collections import defaultdict


LOGGER = log_utils.getLogger()


class OHEAlignmentArbiter(OHEAlignmentBase):
    def __init__(self):
        super(OHEAlignmentArbiter, self).__init__()
        #self.re_encrypt_times = []  # Record the times needed for each host

        self.role = consts.ARBITER
        # self.aggregator = aggregator.Arbiter()
        self.cipher = paillier_cipher.Arbiter()

    def _init_model(self, params):
        super()._init_model(params)
        #self.cipher.register_paillier_cipher(self.transfer_variable)

    def fit(self, data_instances=None):

        """This is used when there is a need for aligment within the
        federated learning. The function would align the column headers from
        guest and host and send the new aligned headers back.

        Args: 
            data_instances: data itself

        
        Returns:
            Combine all the column headers from guest and host 
            if there is alignment is used
        """

        
        # self.init_schema(data_instances)
        # self._init_params(data_instances)
        
        if self.need_alignment:

            self.guest_columns = self.transfer_variable.guest_columns.get(idx = -1) # getting guest column
            self.host_columns = self.transfer_variable.host_columns.get(idx = -1) # getting host column
        
            # let's do alignment here
            # simple method first, we will implement Needleman–Wunsch algo in future, if needed
            all_cols_dict = defaultdict(list)

            #Obtain all the guest headers
            for item in self.guest_columns:
                guest_cols = ast.literal_eval(item)
                for k, v in guest_cols.items():
                    all_cols_dict[k] = list(set(all_cols_dict[k]+v))
            
            #Obtain all the host headers
            for item in self.host_columns:
                host_cols = ast.literal_eval(item)
                for k, v in host_cols.items():
                    all_cols_dict[k] = list(set(all_cols_dict[k]+v))

            #Align all of them together
            combined_all_cols = {}
            for el in all_cols_dict.keys():
                combined_all_cols[el] = all_cols_dict[el]
            
            
            LOGGER.info("**AIFEL** combined cols: {}".format(combined_all_cols))

            #Send the aligned headers back to guest and host
            self.transfer_variable.aligned_columns.remote("{}".format(combined_all_cols), role = consts.HOST, idx = -1)
            self.transfer_variable.aligned_columns.remote("{}".format(combined_all_cols), role = consts.GUEST, idx = -1)



    def _get_meta(self):
        pass

    def _get_param(self):
        pass

    def export_model(self):
        
        return None

    def _load_model(self, model_dict):
        pass

    def transform(self, data_instances):
        pass

    def load_model(self, model_dict):
        pass 