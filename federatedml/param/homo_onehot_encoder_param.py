#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# added by jsweng
# param class for OHE with alignment
#
from federatedml.param.base_param import BaseParam


class HomoOneHotParam(BaseParam):
    """

    Parameters
    ----------

    transform_col_indexes: list or int, default: -1
        Specify which columns need to calculated. -1 represent for all columns.

    need_run: bool, default True
        Indicate if this module needed to be run

    need_alignment: bool, default True
        Indicated whether alignment of features is turned on

    """

    def __init__(self, transform_col_indexes=-1, transform_col_names=None, need_run=True, need_alignment=True):
        super(HomoOneHotParam, self).__init__()
        if transform_col_names is None:
            transform_col_names = []
        self.transform_col_indexes = transform_col_indexes
        self.transform_col_names = transform_col_names
        self.need_run = need_run
        self.need_alignment = need_alignment

    def check(self):
        descr = "One-hot encoder with alignment param's"
        self.check_defined_type(self.transform_col_indexes, descr, ['list', 'int'])
        self.check_boolean(self.need_run, descr)
        self.check_boolean(self.need_alignment, descr)
        return True
