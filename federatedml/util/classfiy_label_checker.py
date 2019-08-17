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
    def validate_y(y):
        """
        Label Checker in classification task.
            Check whether the distinct labels is no more than MAX_CLASSNUM which define in consts,
            also get all distinct lables

        Parameters
        ----------
        y : DTable
            The input data's labels

        Returns
        -------
        num_class : int, the number of distinct labels

        labels : list, the distince labels

        """
        class_dict_iters = y.mapPartitions(ClassifyLabelChecker.get_all_class).collect()
        class_dict = {}
        for _, worker_class_dict in class_dict_iters:
            class_dict.update(worker_class_dict)

        num_class = len(class_dict)
        if len(class_dict) > consts.MAX_CLASSNUM:
            raise ValueError("In Classfy Proble, max dif classes should no more than %d" % (consts.MAX_CLASSNUM))

        return num_class, class_dict.keys()

    @staticmethod
    def get_all_class(kv_iterator):
        class_dict = {}
        for _, label in kv_iterator:
            class_dict[label] = True

        if len(class_dict) > consts.MAX_CLASSNUM:
            raise ValueError("In Classfy Task, max dif classes should no more than %d" % (consts.MAX_CLASSNUM))

        return class_dict


class RegressionLabelChecker(object):
    @staticmethod
    def validate_y(y):
        """
        Label Checker in regression task.
            Check if all labels is a float type.

        Parameters
        ----------
        y : DTable
            The input data's labels

        """
        y.mapValues(RegressionLabelChecker.test_numeric_data)

    @staticmethod
    def test_numeric_data(value):
        try:
            y = float(value)
        except:
            raise ValueError("In Regression Task, all label should be numeric!!")
