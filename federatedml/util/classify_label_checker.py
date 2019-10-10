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
# Lable Cechker
# =============================================================================

from federatedml.util import consts


class ClassifyLabelChecker(object):
    def __init__(self):
        pass

    @staticmethod
    def validate_label(data_inst):
        """
        Label Checker in classification task.
            Check whether the distinct labels is no more than MAX_CLASSNUM which define in consts,
            also get all distinct labels

        Parameters
        ----------
        data_inst : DTable,
                    values are data instance format define in federatedml/feature/instance.py

        Returns
        -------
        num_class : int, the number of distinct labels

        labels : list, the distince labels

        """
        class_set = data_inst.mapPartitions(ClassifyLabelChecker.get_all_class).reduce(lambda x, y: x | y)

        num_class = len(class_set)
        if len(class_set) > consts.MAX_CLASSNUM:
            raise ValueError("In Classfy Proble, max dif classes should no more than %d" % (consts.MAX_CLASSNUM))

        return num_class, list(class_set)

    @staticmethod
    def get_all_class(kv_iterator):
        class_set = set()
        for _, inst in kv_iterator:
            class_set.add(inst.label)

        if len(class_set) > consts.MAX_CLASSNUM:
            raise ValueError("In Classify Task, max dif classes should no more than %d" % (consts.MAX_CLASSNUM))

        return class_set


class RegressionLabelChecker(object):
    @staticmethod
    def validate_label(data_inst):
        """
        Label Checker in regression task.
            Check if all labels is a float type.

        Parameters
        ----------
        data_inst : DTable,
                    values are data instance format define in federatedml/feature/instance.py

        """
        data_inst.mapValues(RegressionLabelChecker.test_numeric_data)

    @staticmethod
    def test_numeric_data(value):
        try:
            label = float(value.label)
        except:
            raise ValueError("In Regression Task, all label should be numeric!!")
