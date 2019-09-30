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
import time

from arch.api.model_manager import manager as model_manager
from arch.api.utils import log_utils
from federatedml.protobuf.generated import one_vs_rest_param_pb2
from federatedml.transfer_variable.transfer_class.one_vs_rest_transfer_variable import OneVsRestTransferVariable
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class OneVsRest(object):
    def __init__(self, classifier, role, mode, one_vs_rest_param):
        self.classifier = classifier
        self.transfer_variable = OneVsRestTransferVariable()

        self.classes = None
        self.role = role
        self.mode = mode
        self.flow_id = 0
        self.has_arbiter = one_vs_rest_param.has_arbiter

        self.models = []
        self.class_name = self.__class__.__name__
        self.__support_role_list = [consts.GUEST, consts.HOST, consts.ARBITER]
        self.__support_mode_list = [consts.HOMO, consts.HETERO]

    # @staticmethod
    # def __mask_label(instance, label):
    #     instance.label = (1 if (instance.label == label) else 0)
    #     return instance

    # @staticmethod
    # def __get_classes(data_instances):
    #     classes = set()
    #     for instance in data_instances:
    #         classes.add(instance[1].label)
    #
    #     return classes

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

    def _get_data_classes(self, data_instances):
        """
        get all classes in data_instances
        """
        def get_classes(data_instances):
            classes = set()
            for instance in data_instances:
                classes.add(instance[1].label)

            return classes

        f = functools.partial(get_classes)
        classes_res = data_instances.mapPartitions(f)
        classes = classes_res.reduce(lambda a, b: a | b)
        return classes

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

    def fit(self, data_instances=None):
        """
        Fit OneVsRest model
        Parameters:
        ----------
        data_instances: DTable of instances
        """
        LOGGER.info("mode is {}, role is {}, start to one_vs_rest fit".format(self.mode, self.role))
        if self.has_label:
            class_set = self._get_data_classes(data_instances)
        else:
            class_set = None

        LOGGER.info("Start to synchronize")
        self._synchronize_classes_list(class_set)

        LOGGER.info("Total classes:{}".format(self.classes))

        current_flow_id = self.classifier.flowid
        for label_index, label in enumerate(self.classes):
            LOGGER.info("Start to train OneVsRest with label_index:{}, label:{}".format(label_index, label))
            classifier = copy.deepcopy(self.classifier)
            classifier.set_flowid("_".join([current_flow_id, "one_vs_rest", str(label_index)]))
            if self.has_label:
                header = data_instances.schema.get("header")
                data_instances_mask_label = self._mask_data_label(data_instances, label=label)
                data_instances_mask_label.schema['header'] = header
                LOGGER.info("finish mask label:{}".format(label))

                LOGGER.info("start classifier fit")
                classifier.fit(data_instances_mask_label)
            else:
                LOGGER.info("start classifier fit")
                classifier.fit(data_instances)

            self.models.append(classifier)
            LOGGER.info("Finish model_{} training!".format(label_index))

    def predict(self, data_instances):
        """
        Predict OneVsRest model
        Parameters:
        ----------
        data_instances: DTable of instances
        predict_param: PredictParam of classifier

        Returns:
        ----------
        predict_res: DTable, if has predict_res, it includes ground true label, predict probably and predict label
        """
        prob = None
        for i, model in enumerate(self.models):
            current_flow_id = model.flowid
            model.set_flowid("_".join([current_flow_id, "one_vs_rest", str(i)]))

            LOGGER.info("Start to predict with model:{}".format(i))
            # model.set_flowid("predict_" + str(i))
            predict_res = model.predict(data_instances)
            if predict_res:
                if not prob:
                    prob = predict_res.mapValues(lambda r: [r[2]])
                else:
                    prob = prob.join(predict_res, lambda p, r: p + [r[2]])

            LOGGER.info("finish model_{} predict.".format(i))

        predict_res = None
        if prob:
            f = functools.partial(self.__get_multi_class_res, classes=list(self.classes))
            multi_classes_res = prob.mapValues(f)
            predict_res = data_instances.join(multi_classes_res, lambda d, m: [d.label, m[0], m[1], m[2]])

        LOGGER.info("finish OneVsRest Predict, return predict results.")

        return predict_res

    def save_model(self, name, namespace):
        """
        Save each classifier model of OneVsRest. It just include model_param but not model_meta now
        """
        classifier_models = []
        str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        for i, model in enumerate(self.models):
            classifier_name = str_time + "_" + str(i) + "_" + self.role + "_name"
            model.save_model(classifier_name, namespace)
            classifier_model = one_vs_rest_param_pb2.ClassifierModel(name=classifier_name,
                                                                     namespace=namespace)
            classifier_models.append(classifier_model)
            LOGGER.info("finish save model_{}, role:{}".format(i, self.role))

        str_classes = [str(c) for c in self.classes]
        one_vs_rest_param_obj = one_vs_rest_param_pb2.OneVsRestParam(classes=str_classes,
                                                                     classifier_models=classifier_models)

        param_buffer_type = "{}.param".format(self.class_name)
        model_manager.save_model(buffer_type=param_buffer_type,
                                 proto_buffer=one_vs_rest_param_obj,
                                 name=name,
                                 namespace=namespace)

        meta_buffer_type = 'None'

        LOGGER.info("finish OneVsRest save model.")
        return [(meta_buffer_type, param_buffer_type)]

    def load_model(self, name, namespace):
        """
        Load OneVsRest model
        """
        model_obj = one_vs_rest_param_pb2.OneVsRestParam()
        buffer_type = "{}.param".format(self.class_name)

        model_manager.read_model(buffer_type=buffer_type,
                                 proto_buffer=model_obj,
                                 name=name,
                                 namespace=namespace)

        LOGGER.info("OneVsRest classes number:{}".format(len(model_obj.classes)))

        self.models = []
        for i, classifier_model in enumerate(model_obj.classifier_models):
            classifier = copy.deepcopy(self.classifier)
            classifier.load_model(classifier_model.name, classifier_model.namespace)
            self.models.append(classifier)
            LOGGER.info("finish load model_{}, classes is {}".format(i, model_obj.classes[i]))

        self.classes = []
        for model_class in model_obj.classes:
            self.classes.append(model_class)

        LOGGER.info("finish load OneVsRest model.")


class HomoOneVsRest(OneVsRest):
    # def __init__(self, classifier, role, one_vs_rest_param):
    #     super().__init__()

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
        class_nums = self.transfer_variable.aggregate_classes.get(idx=0)
        self.classes = [x for x in range(class_nums)]

    def _sync_class_arbiter(self):
        class_nums = self.transfer_variable.aggregate_classes.get(idx=0)
        self.classes = [x for x in range(class_nums)]


def one_vs_rest_factory(classifier, role, mode, one_vs_rest_param):
    if mode == consts.HOMO:
        return HomoOneVsRest(classifier, role, mode, one_vs_rest_param)
    elif mode == consts.HETERO:
        return HeteroOneVsRest(classifier, role, mode, one_vs_rest_param)
    else:
        raise ValueError(f"Cannot recognize mode: {mode} in one vs rest")
