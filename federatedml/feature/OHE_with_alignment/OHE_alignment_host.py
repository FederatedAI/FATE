#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  added by jsweng
#  alignemnet host

import numpy as np

from arch.api.utils import log_utils
from federatedml.framework.homo.procedure import aggregator
from federatedml.framework.homo.procedure import paillier_cipher
from federatedml.feature.OHE_with_alignment.OHE_alignment_base import OHEAlignmentBase
from federatedml.util import consts
import functools
import ast
from collections import defaultdict


LOGGER = log_utils.getLogger()


class OHEAlignmentHost(OHEAlignmentBase):
    def __init__(self):
        super(OHEAlignmentHost, self).__init__()
        self.role = consts.HOST
        
    def _init_model(self, params):
        super()._init_model(params)

    def fit(self, data_instances):

        """This  allows for one-hot-encoding of the 
        columns with or without alignment with the other parties
        in the federated learning.

        Args:
            data_instances: data the host has access to

        Returns:
            if alignment is on, then the one-hot-encoding data_instances are done with
            alignment with parties involved in federated learning else,
            the data is one-hot-encoded independently

        """


        self.init_schema(data_instances)
        self._init_params(data_instances)
        ori_header = self.inner_param.header.copy() # keep a copy of original header

        #obtain the individual column headers with their values
        f1 = functools.partial(self.record_new_header,
                               inner_param=self.inner_param)
        self.col_maps = data_instances.mapPartitions(f1).reduce(self.merge_col_maps)
        LOGGER.debug("**AIFEL** new col_maps is: {}".format(self.col_maps))
        col_maps = {}
        for col_name, pair_obj in self.col_maps.items():
            values = [str(x) for x in pair_obj.values]            
            col_maps[col_name] = values        
        LOGGER.debug("**AIFEL** flattened new col_maps is: {}".format(col_maps))

        #feature alignment is executed in a 'switch' manner
        if self.need_alignment: 

            # send to arbiter
            self.transfer_variable.host_columns.remote("{}".format(col_maps), role = consts.ARBITER, idx = -1) # send guest size to arbiter
            LOGGER.debug("**AIFEL** host column sent")
            
            ## receive aligned columns from arbiter
            aligned_columns = self.transfer_variable.aligned_columns.get(idx=-1)
            aligned_col_maps = ast.literal_eval(aligned_columns[0])
            LOGGER.debug("**AIFEL** aligned columns received are: {}".format(aligned_col_maps))

            ## transform with aligned columns
            ## ['age', 'sex_0', 'sex_1', 'cp_1', 'cp_2', 'cp_3', 'cp_4', 'trestbps', 'chol', 'fbs_0', 'fbs_1', 'restecg_0', 'restecg_1', 'restecg_2', 'thalach', 'exang_0', 'exang_1', 'oldpeak', 'slope_1', 'slope_2', 'slope_3', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'thal_3', 'thal_6', 'thal_7']"
            
            #All the headers - original or not - are appended together
            new_header = []
            transform_col_names = []
            for col in ori_header: 
                if col not in aligned_col_maps:
                    new_header.append(col)
                    continue
                transform_col_names.append(col)
                for vv in aligned_col_maps[col]:
                    new_header.append(col+'_'+vv)
            
            LOGGER.debug("**AIFEL** new transform col names after format received aligned columns: {}".format(transform_col_names))        
            LOGGER.debug("**AIFEL** new header after format received aligned columns: {}".format(new_header))

            self.inner_param.add_transform_names = transform_col_names
            self.inner_param.set_result_header(new_header)

            LOGGER.debug("Before set_schema in fit, schema is : {}, header: {}".format(self.schema,
                                                                                    self.inner_param.header))
        
        else:
            self._transform_schema() #Calling the original function
        
        data_instances = self.transform(data_instances)
        LOGGER.debug("After transform in fit, schema is : {}, header: {}".format(self.schema,
                                                                                 self.inner_param.header))
        # LOGGER.debug("After transform the meta is: {}".format(dict(self._get_param().col_map)))

        
        return data_instances