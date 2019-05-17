import copy
import functools
import time

from arch.api import federation
from arch.api.model_manager import manager as model_manager
from arch.api.proto import one_vs_rest_param_pb2
from arch.api.utils import log_utils
from federatedml.evaluation import Evaluation
from federatedml.util import consts
from federatedml.util.transfer_variable import OneVsRestTransferVariable
from federatedml.param.param import PredictParam

LOGGER = log_utils.getLogger()


class OneVsRest(object):
    def __init__(self, classifier, role, mode):
        self.classifier = classifier
        self.transfer_variable = OneVsRestTransferVariable()

        self.num_classes = None
        self.classes = None
        self.role = role
        self.mode = mode
        self.flow_id = 0
        self.need_mask_label = False

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
    def __get_classes(data_instances):
        classes = set()
        for instance in data_instances:
            classes.add(instance[1].label)

        return classes

    def __get_data_classes(self, data_instances):
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

    @staticmethod
    def __mask_label(instance, label):
        instance.label = (1 if (instance.label == label) else 0)

        return instance

    def __mask_data_label(self, data_instances, label):
        f = functools.partial(self.__mask_label, label=label)
        data_instances = data_instances.mapValues(f)

        return data_instances

    def __synchronize_aggregate_classed_list(self):
        if self.role == consts.GUEST:
            federation.remote(self.classes,
                              name=self.transfer_variable.aggregate_classes.name,
                              tag=self.transfer_variable.generate_transferid(self.transfer_variable.aggregate_classes),
                              role=consts.HOST,
                              idx=0)

            federation.remote(self.classes,
                              name=self.transfer_variable.aggregate_classes.name,
                              tag=self.transfer_variable.generate_transferid(self.transfer_variable.aggregate_classes),
                              role=consts.ARBITER,
                              idx=0)

        elif self.role == consts.HOST or self.role == consts.ARBITER:
            self.classes = federation.get(name=self.transfer_variable.aggregate_classes.name,
                                          tag=self.transfer_variable.generate_transferid(
                                              self.transfer_variable.aggregate_classes),
                                          idx=0)

        else:
            raise ValueError("Unknown role:{}".format(self.role))

    def __synchronize_classes_list__synchronize_classes_list(self):
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

    def fit(self, data_instances=None):
        if (self.mode == consts.HOMO and self.role != consts.ARBITER) or (self.mode == consts.HETERO and self.role == consts.GUEST):
            LOGGER.info("mode is {}, role is {}, start to get data classes".format(self.mode, self.role))
            self.classes = self.__get_data_classes(data_instances)
            self.need_mask_label = True
            LOGGER.debug("classes:{}".format(self.classes))

        LOGGER.info("Start to synchronize")
        self.__synchronize_classes_list()

        LOGGER.info("Total classes:{}".format(self.classes))
        self.num_classes = len(self.classes)

        for flow_id, label in enumerate(self.classes):
            LOGGER.info("Start to train OneVsRest with flow_id:{}, label:{}".format(flow_id, label))
            classifier = copy.copy(self.classifier)
            classifier.set_flowid("train_" + str(flow_id))
            if self.need_mask_label:
                header = data_instances.schema.get("header")
                data_instances_mask_label = self.__mask_data_label(data_instances, label=label)
                data_instances_mask_label.schema['header'] = header
                LOGGER.debug("finish mask label:{}".format(label))

                LOGGER.debug("start classifier fit")
                classifier.fit(data_instances_mask_label)
            else:
                LOGGER.debug("start classifier fit")
                classifier.fit(data_instances)

            self.models.append(classifier)
            LOGGER.info("Finish model_{} training!".format(flow_id))

    @staticmethod
    def __get_multi_class_res(instance):
        max_prob = -1
        max_prob_index = -1
        for (i, prob) in enumerate(instance):
            if prob > max_prob:
                max_prob = prob
                max_prob_index = i

        return max_prob, max_prob_index

    @staticmethod
    def __append(list_obj, value):
        list_obj.append(value)
        return list_obj

    def predict(self, data_instances):
        prob = None
        for i, model in enumerate(self.models):
            LOGGER.info("Start to predict with model:{}".format(i))
            predict_param = PredictParam()
            model.set_flowid("predict_" + str(i))
            predict_res = model.predict(data_instances, predict_param)
            if predict_res:
                if not prob:
                    prob = predict_res.mapValues(lambda r: [r[1]])
                else:
                    f = functools.partial(self.__append)
                    prob = prob.join(predict_res, lambda p, r: f(list_obj=p, value=r[1]))

            LOGGER.info("finish model_{} predict.".format(i))

        multi_classes_res = None
        if prob:
            f = functools.partial(self.__get_multi_class_res)
            multi_classes_res = prob.mapValues(f)



        LOGGER.info("finish OneVsRest Predict, return predict results.")

        return multi_classes_res

    def save_model(self, name, namespace):
        classifier_models = []
        for i, model in enumerate(self.models):
            str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            classifier_name = str_time + "_" + str(i) + "_" + self.role + "_name"
            classifier_namespace = str_time + "_" + str(i) + "_" + self.role + "_namespace"

            model.save_model(name, namespace)
            classifier_model = one_vs_rest_param_pb2.ClassifierModel(name=classifier_name, namespace=classifier_namespace)
            classifier_models.append(classifier_model)
            LOGGER.debug("finish save model_{}, role:{}".format(i, self.role))

        str_classes = [ str(c) for c in self.classes]
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
        model_obj = one_vs_rest_param_pb2.OneVsRestParam()
        buffer_type = "{}.param".format(self.class_name)

        model_manager.read_model(buffer_type=buffer_type,
                                 proto_buffer=model_obj,
                                 name=name,
                                 namespace=namespace)

        LOGGER.debug("model_obj:{}".format(model_obj))

        LOGGER.info("OneVsRest classes number:{}".format(len(model_obj.classes)))

        self.models = []
        for i, classifier_model in enumerate(model_obj.classifier_models):
            classifier = copy.copy(self.classifier)
            classifier.load_model(classifier_model.name, classifier_model.namespace)
            self.models.append(classifier)
            LOGGER.info("finish load model_{}, classes is {}".format(i, model_obj.classes[i]))

        LOGGER.info("finish load OneVsRest model.")


    def evaluate(self, labels, pred_prob, pred_labels, evaluate_param):
        # predict_res = None
        # if evaluate_param.classi_type == consts.BINARY:
        #     predict_res = pred_prob
        # elif evaluate_param.classi_type == consts.MULTY:
        #     predict_res = pred_labels
        # else:
        #     LOGGER.warning("unknown classification type, return None as evaluation results")

        eva = Evaluation(evaluate_param.classi_type)
        return eva.report(labels, predict_res, evaluate_param.metrics, evaluate_param.thresholds,
                          evaluate_param.pos_label)