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
import numpy as np
import time

from arch.api import federation
from arch.api.model_manager import manager as model_manager
from arch.api.proto import one_vs_rest_param_pb2
from arch.api.utils import log_utils
from federatedml.evaluation import Evaluation
from federatedml.util import consts
from federatedml.util.transfer_variable import OneVsRestTransferVariable

LOGGER = log_utils.getLogger()



class OneVsRest(object):
    def __init__(self, classifier, role, mode, one_vs_rest_param):
        self.classifier = classifier
        self.transfer_variable = OneVsRestTransferVariable()

        self.classes = None
        self.role = role
        self.mode = mode
        self.flow_id = 0
        self.need_mask_label = False
        self.has_arbiter = one_vs_rest_param.has_arbiter

        self.models = []
        self.class_name = self.__class__.__name__
        self.__support_role_list = [consts.GUEST, consts.HOST, consts.ARBITER]
        self.__support_mode_list = [consts.HOMO, consts.HETERO]

    def __check_param(self):
        if self.role not in self.__support_role_list:
            raise ValueError("Not support role:{}".format(self.role))

        if self.mode not in self.__support_mode_list:
            raise ValueError("Not support mode:{}".format(self.mode))

    @staticmethod
    def __mask_label(instance, label):
        instance.label = (1 if (instance.label == label) else 0)

        return instance

    @staticmethod
    def __get_classes(data_instances):
        classes = set()
        for instance in data_instances:
            classes.add(instance[1].label)

        return classes

    @staticmethod
    def __get_multi_class_res(instance, classes):
        """
        return max_prob and its class where max_prob is the max probably in input instance
        """
        max_prob = -1
        max_prob_index = -1
        for (i, prob) in enumerate(instance):
            if prob > max_prob:
                max_prob = prob
                max_prob_index = i

        return max_prob, classes[max_prob_index]

    @staticmethod
    def __append(list_obj, value):
        list_obj.append(value)
        return list_obj

    def __get_data_classes(self, data_instances):
        """
        get all classes in data_instances
        """
        f = functools.partial(self.__get_classes)
        classes_res = data_instances.mapPartitions(f)

        classes = None
        for res in list(classes_res.collect()):
            class_set = res[1]
            if not classes:
                classes = class_set

            for v in class_set:
                classes.add(v)

        return classes

    def __mask_data_label(self, data_instances, label):
        """
        mask the instance.label to 1 if equals to label and 0 if not
        """
        f = functools.partial(self.__mask_label, label=label)
        data_instances = data_instances.mapValues(f)

        return data_instances

    def __synchronize_aggregate_classed_list(self):
        """
        synchronize all of class of data, include guest, host and arbiter, from guest to the others
        """
        if self.role == consts.GUEST:
            federation.remote(self.classes,
                              name=self.transfer_variable.aggregate_classes.name,
                              tag=self.transfer_variable.generate_transferid(self.transfer_variable.aggregate_classes),
                              role=consts.HOST,
                              idx=0)

            if self.has_arbiter:
                federation.remote(self.classes,
                                  name=self.transfer_variable.aggregate_classes.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.aggregate_classes),
                                  role=consts.ARBITER,
                                  idx=0)

        elif self.role == consts.HOST or self.role == consts.ARBITER:
            self.classes = federation.get(name=self.transfer_variable.aggregate_classes.name,
                                          tag=self.transfer_variable.generate_transferid(
                                              self.transfer_variable.aggregate_classes),
                                          idx=0)

        else:
            raise ValueError("Unknown role:{}".format(self.role))

    def __synchronize_classes_list(self):
        """
        Guest will get classes from host data, and aggregate classes it has. After that, send the aggregate classes to
        host and arbiter as binary classification times.
        """
        if self.mode == consts.HOMO:
            if self.role == consts.GUEST:
                host_classes_list = federation.get(name=self.transfer_variable.host_classes.name,
                                                   tag=self.transfer_variable.generate_transferid(
                                                       self.transfer_variable.host_classes),
                                                   idx=0)

                for host_class in host_classes_list:
                    self.classes.add(host_class)

            elif self.role == consts.HOST:
                federation.remote(self.classes,
                                  name=self.transfer_variable.host_classes.name,
                                  tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_classes),
                                  role=consts.GUEST,
                                  idx=0)

        self.__synchronize_aggregate_classed_list()

    def set_flowid(self, flowid=0):
        """
        Set the flowid of each classifier, because each classifier should has different flowid
        """
        if self.transfer_variable is not None:
            self.transfer_variable.set_flowid(flowid)
            LOGGER.info("set flowid:" + str(flowid))

    def fit(self, data_instances=None):
        """
        Fit OneVsRest model
        Parameters:
        ----------
        data_instances: DTable of instances
        """
        if (self.mode == consts.HOMO and self.role != consts.ARBITER) or (
                self.mode == consts.HETERO and self.role == consts.GUEST):
            LOGGER.info("mode is {}, role is {}, start to get data classes".format(self.mode, self.role))
            self.classes = self.__get_data_classes(data_instances)
            self.need_mask_label = True

        LOGGER.info("Start to synchronize")
        self.__synchronize_classes_list()

        LOGGER.info("Total classes:{}".format(self.classes))

        for flow_id, label in enumerate(self.classes):
            LOGGER.info("Start to train OneVsRest with flow_id:{}, label:{}".format(flow_id, label))
            classifier = copy.deepcopy(self.classifier)
            classifier.set_flowid("train_" + str(flow_id))
            if self.need_mask_label:
                header = data_instances.schema.get("header")
                data_instances_mask_label = self.__mask_data_label(data_instances, label=label)
                data_instances_mask_label.schema['header'] = header
                LOGGER.info("finish mask label:{}".format(label))

                LOGGER.info("start classifier fit")
                classifier.fit(data_instances_mask_label)
            else:
                LOGGER.info("start classifier fit")
                classifier.fit(data_instances)

            self.models.append(classifier)
            LOGGER.info("Finish model_{} training!".format(flow_id))

    def predict(self, data_instances, predict_param):
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
            LOGGER.info("Start to predict with model:{}".format(i))
            model.set_flowid("predict_" + str(i))
            predict_res = model.predict(data_instances, predict_param)
            if predict_res:
                if not prob:
                    prob = predict_res.mapValues(lambda r: [r[1]])
                else:
                    f = functools.partial(self.__append)
                    prob = prob.join(predict_res, lambda p, r: f(list_obj=p, value=r[1]))

            LOGGER.info("finish model_{} predict.".format(i))

        predict_res = None
        if prob:
            f = functools.partial(self.__get_multi_class_res, classes=list(self.classes))
            multi_classes_res = prob.mapValues(f)
            predict_res = data_instances.join(multi_classes_res, lambda d, m: (d.label, m[0], m[1]))

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

    def evaluate(self, labels, pred_prob, pred_labels, evaluate_param):
        """
        evaluate OneVsRest model.
        Parameters:
        ----------
        labels: list, ground true label
        pred_prob: list, predict probably of pred_labels
        pred_labels: list, predict label
        evaluate_param: EvaluateParam
        Returns:
        ----------
        evaluate results
        """
        predict_res = None
        if evaluate_param.classi_type == consts.BINARY:
            predict_res = pred_prob
        elif evaluate_param.classi_type == consts.MULTY:
            predict_res = pred_labels
        else:
            LOGGER.warning("unknown classification type, return None as evaluation results")

        eva = Evaluation(evaluate_param.classi_type)

        label_type = type(labels[0])

        if isinstance(predict_res, np.ndarray) and isinstance(labels, np.ndarray):
            predict_res = predict_res.astype(labels.dtype)
        else:
            if not isinstance(predict_res, list):
                predict_res = list(predict_res)

            for i in range(len(predict_res)):
                predict_res[i] = label_type(predict_res[i])

        return eva.report(labels, predict_res, evaluate_param.metrics, evaluate_param.thresholds,
                          evaluate_param.pos_label)
