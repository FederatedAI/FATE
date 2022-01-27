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

import copy
import functools

from federatedml.feature.instance import Instance
from federatedml.transfer_variable.transfer_class.one_vs_rest_transfer_variable import OneVsRestTransferVariable
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util.classify_label_checker import ClassifyLabelChecker
from federatedml.util.io_check import assert_io_num_rows_equal


class OneVsRest(object):
    def __init__(self, classifier, role, mode, has_arbiter):
        self.classifier = classifier
        self.transfer_variable = OneVsRestTransferVariable()

        self.classes = None
        self.role = role
        self.mode = mode
        self.flow_id = 0
        self.has_arbiter = has_arbiter

        self.models = []
        self.class_name = self.__class__.__name__

    @staticmethod
    def __get_multi_class_res(instance, classes):
        """
        return max_prob and its class where max_prob is the max probably in input instance
        """
        max_prob = -1
        max_prob_index = -1
        instance_with_class = {}
        for (i, prob) in enumerate(instance):
            instance_with_class[classes[i]] = prob
            if prob > max_prob:
                max_prob = prob
                max_prob_index = i

        return classes[max_prob_index], max_prob, instance_with_class

    def get_data_classes(self, data_instances):
        """
        get all classes in data_instances
        """
        class_set = None
        if self.has_label:
            num_class, class_list = ClassifyLabelChecker.validate_label(data_instances)
            class_set = set(class_list)
        self._synchronize_classes_list(class_set)
        return self.classes

    @staticmethod
    def _mask_data_label(data_instances, label):
        """
        mask the instance.label to 1 if equals to label and 0 if not
        """

        def do_mask_label(instance):
            instance.label = (1 if (instance.label == label) else 0)
            return instance

        f = functools.partial(do_mask_label)
        data_instances = data_instances.mapValues(f)

        return data_instances

    def _sync_class_guest(self, class_set):
        raise NotImplementedError("Function should not be called here")

    def _sync_class_host(self, class_set):
        raise NotImplementedError("Function should not be called here")

    def _sync_class_arbiter(self):
        raise NotImplementedError("Function should not be called here")

    def _synchronize_classes_list(self, class_set):
        """
        Guest will get classes from host data, and aggregate classes it has. After that, send the aggregate classes to
        host and arbiter as binary classification times.
        """
        if self.role == consts.GUEST:
            self._sync_class_guest(class_set)
        elif self.role == consts.HOST:
            self._sync_class_host(class_set)
        else:
            self._sync_class_arbiter()

    @property
    def has_label(self):
        raise NotImplementedError("Function should not be called here")

    def fit(self, data_instances=None, validate_data=None):
        """
        Fit OneVsRest model
        Parameters:
        ----------
        data_instances: Table of instances
        """

        LOGGER.info("mode is {}, role is {}, start to one_vs_rest fit".format(self.mode, self.role))

        LOGGER.info("Total classes:{}".format(self.classes))

        self.classifier.callback_one_vs_rest = True
        current_flow_id = self.classifier.flowid
        summary_dict = {}
        for label_index, label in enumerate(self.classes):
            LOGGER.info("Start to train OneVsRest with label_index:{}, label:{}".format(label_index, label))
            classifier = copy.deepcopy(self.classifier)
            classifier.need_one_vs_rest = False
            classifier.set_flowid(".".join([current_flow_id, "model_" + str(label_index)]))
            if self.has_label:
                header = data_instances.schema.get("header")
                data_instances_mask_label = self._mask_data_label(data_instances, label=label)
                data_instances_mask_label.schema['header'] = header

                if validate_data is not None:
                    validate_mask_label_data = self._mask_data_label(validate_data, label=label)
                    validate_mask_label_data.schema['header'] = header
                else:
                    validate_mask_label_data = validate_data
                LOGGER.info("finish mask label:{}".format(label))

                LOGGER.info("start classifier fit")
                classifier.fit_binary(data_instances_mask_label, validate_data=validate_mask_label_data)
            else:
                LOGGER.info("start classifier fit")
                classifier.fit_binary(data_instances, validate_data=validate_data)
            _summary = classifier.summary()
            _summary['one_vs_rest'] = True
            summary_dict[label] = _summary
            self.models.append(classifier)
            if hasattr(self, "header"):
                header = getattr(self, "header")
                if header is None:
                    setattr(self, "header", getattr(classifier, "header"))
            LOGGER.info("Finish model_{} training!".format(label_index))
        self.classifier.set_summary(summary_dict)

    def _comprehensive_result(self, predict_res_list):
        """
        prob result is available for guest party only.
        """
        if self.role == consts.GUEST:
            # assert 1 == 2, f"predict_res_list: {predict_res_list[0].first()[1].features}"
            prob = predict_res_list[0].mapValues(lambda r: [r.features[2]])
            for predict_res in predict_res_list[1:]:
                prob = prob.join(predict_res, lambda p, r: p + [r.features[2]])
        else:
            prob = None
        return prob

    @assert_io_num_rows_equal
    def predict(self, data_instances):
        """
        Predict OneVsRest model
        Parameters:
        ----------
        data_instances: Table of instances
        predict_param: PredictParam of classifier

        Returns:
        ----------
        predict_res: Table, if has predict_res, it includes ground true label, predict probably and predict label
        """
        LOGGER.info("Start one_vs_all predict procedure.")
        predict_res_list = []
        for i, model in enumerate(self.models):
            current_flow_id = model.flowid
            model.set_flowid(".".join([current_flow_id, "model_" + str(i)]))

            LOGGER.info("Start to predict with model:{}".format(i))
            # model.set_flowid("predict_" + str(i))
            single_predict_res = model.predict(data_instances)
            predict_res_list.append(single_predict_res)

        prob = self._comprehensive_result(predict_res_list)
        if prob:
            f = functools.partial(self.__get_multi_class_res, classes=list(self.classes))
            multi_classes_res = prob.mapValues(f)
            predict_res = data_instances.join(multi_classes_res, lambda d, m: [d.label, m[0], m[1], m[2]])

            def _transfer(instance, pred_res):
                return Instance(features=pred_res, inst_id=instance.inst_id)

            predict_res = data_instances.join(predict_res, _transfer)
        else:
            predict_res = None
        #
        # LOGGER.info("finish OneVsRest Predict, return predict results.")

        return predict_res

    def save(self, single_model_pb):
        """
        Save each classifier model of OneVsRest. It just include model_param but not model_meta now
        """
        classifier_pb_objs = []
        for classifier in self.models:
            single_param_dict = classifier.get_single_model_param()
            classifier_pb_objs.append(single_model_pb(**single_param_dict))

        one_vs_rest_class = [str(x) for x in self.classes]
        one_vs_rest_result = {
            'completed_models': classifier_pb_objs,
            'one_vs_rest_classes': one_vs_rest_class
        }
        return one_vs_rest_result

    def load_model(self, one_vs_rest_result):
        """
        Load OneVsRest model
        """
        completed_models = one_vs_rest_result.completed_models
        one_vs_rest_classes = one_vs_rest_result.one_vs_rest_classes
        self.classes = [int(x) for x in one_vs_rest_classes]  # Support other label type in the future
        self.models = []
        for classifier_obj in list(completed_models):
            classifier = copy.deepcopy(self.classifier)
            classifier.load_single_model(classifier_obj)
            classifier.need_one_vs_rest = False
            self.models.append(classifier)


class HomoOneVsRest(OneVsRest):

    def __init__(self, classifier, role, mode, has_arbiter):
        super().__init__(classifier, role, mode, has_arbiter)
        self.header = None

    def set_header(self, header):
        self.header = header

    @property
    def has_label(self):
        if self.role == consts.ARBITER:
            return False
        return True

    def _sync_class_guest(self, class_set):
        host_classes_list = self.transfer_variable.host_classes.get(idx=-1)
        for host_class in host_classes_list:
            class_set = class_set | host_class
        self.classes = list(class_set)
        self.transfer_variable.aggregate_classes.remote(self.classes,
                                                        role=consts.HOST,
                                                        idx=-1)
        if self.has_arbiter:
            class_num = len(self.classes)
            self.transfer_variable.aggregate_classes.remote(class_num,
                                                            role=consts.ARBITER,
                                                            idx=0)

    def _sync_class_host(self, class_set):
        self.transfer_variable.host_classes.remote(class_set,
                                                   role=consts.GUEST,
                                                   idx=0)
        self.classes = self.transfer_variable.aggregate_classes.get(idx=0)

    def _sync_class_arbiter(self):
        class_nums = self.transfer_variable.aggregate_classes.get(idx=0)
        self.classes = [x for x in range(class_nums)]


class HeteroOneVsRest(OneVsRest):
    @property
    def has_label(self):
        if self.role == consts.GUEST:
            return True
        return False

    def _sync_class_guest(self, class_set):
        self.classes = list(class_set)
        class_num = len(self.classes)
        self.transfer_variable.aggregate_classes.remote(class_num,
                                                        role=consts.HOST,
                                                        idx=-1)
        if self.has_arbiter:
            self.transfer_variable.aggregate_classes.remote(class_num,
                                                            role=consts.ARBITER,
                                                            idx=0)

    def _sync_class_host(self, class_set):
        LOGGER.debug("Start to get aggregate classes")
        class_nums = self.transfer_variable.aggregate_classes.get(idx=0)
        self.classes = [x for x in range(class_nums)]

    def _sync_class_arbiter(self):
        class_nums = self.transfer_variable.aggregate_classes.get(idx=0)
        self.classes = [x for x in range(class_nums)]


def one_vs_rest_factory(classifier, role, mode, has_arbiter):
    LOGGER.info("Create one_vs_rest object, role: {}, mode: {}".format(role, mode))
    if mode == consts.HOMO:
        return HomoOneVsRest(classifier, role, mode, has_arbiter)
    elif mode == consts.HETERO:
        return HeteroOneVsRest(classifier, role, mode, has_arbiter)
    else:
        raise ValueError(f"Cannot recognize mode: {mode} in one vs rest")
