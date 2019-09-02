# #
# #  Copyright 2019 The FATE Authors. All Rights Reserved.
# #
# #  Licensed under the Apache License, Version 2.0 (the "License");
# #  you may not use this file except in compliance with the License.
# #  You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# #  Unless required by applicable law or agreed to in writing, software
# #  distributed under the License is distributed on an "AS IS" BASIS,
# #  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #  See the License for the specific language governing permissions and
# #  limitations under the License.
# #
#
# import functools
# import re
#
# import numpy as np
# import tensorflow as tf
#
# from arch.api.proto.tf_model_param_pb2 import TFModelParam, Tensor
# from arch.api.utils import log_utils
# from federatedml.homo.weights import TransferableParameters
# from federatedml.model_selection.mini_batch import MiniBatch
#
# LOGGER = log_utils.getLogger()
#
#
# class Model(object):
#
#     def get_model_weights(self) -> TransferableParameters:
#         """
#         get model weights(maybe for transfer)
#         :return: model weights
#         """
#         pass
#
#     def get_gradient_weights(self, batch) -> TransferableParameters:
#         """
#         get gradient weights
#         :return: gradients
#         """
#         pass
#
#     def assign_model_weights(self, weights: TransferableParameters):
#         """
#         assign model weights
#         :param weights:
#         :return:
#         """
#         pass
#
#     def apply_gradient_weights(self, weights: TransferableParameters):
#         """
#         apply gradient weights to optimizer
#         :param weights:
#         :return:
#         """
#         pass
#
#     def evaluate(self, data_instances):
#         pass
#
#     def save_model(self):
#         pass
#
#     def load_model(self):
#         pass
#
#     def train_local_batch(self, batch):
#         """
#         training local models for one batch
#         :param batch: local training data
#         """
#         pass
#
#     def get_batch_accuracy(self, batch):
#         pass
#
#     def train_local(self, data_instances):
#         pass
#
#     def check_additional_params(self, params):
#         pass
#
#     def set_additional_params(self, **params):
#         self.check_additional_params(params)
#         for k, v in params.items():
#             self.__setattr__(k, v)
#
#
# class TFModel(Model):
#     def __init__(self, optimizer, loss, accuracy, train_inputs, predict_inputs,
#                  predict_outputs, train_feed_dict_fn, eval_feed_dict_fn):
#
#         self._optimizer = optimizer
#         self._loss = loss
#         self._accuracy = accuracy
#         self._train_feed_dict_fn = train_feed_dict_fn
#         self._eval_feed_dict_fn = eval_feed_dict_fn
#
#         self._train_inputs = train_inputs
#         self._predict_inputs = predict_inputs
#         self._predict_outputs = predict_outputs
#
#         # insert placeholder to fetch gradients or apply gradients as we need.
#         gradient = self._optimizer.compute_gradients(self._loss)
#         variables_with_grad = [v for g, v in gradient if g is not None]  # todo: remove this
#         self._gradient = self._optimizer.compute_gradients(self._loss, variables_with_grad)
#         self._grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v) for (g, v) in self._gradient]
#         self._apply_grad_op = self._optimizer.apply_gradients(self._grads_holder)
#         self._train_op = self._optimizer.minimize(self._loss)
#
#         self._gradient_variables = [v for v in variables_with_grad]  # variables with gradient
#         self._model_variable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)  # variables in model
#
#         self.sess = tf.Session()
#         self.sess.run(tf.global_variables_initializer())
#         self.num_batch = None
#         self.batch_size = None
#
#         self.saver = tf.train.Saver()
#
#     def get_model_weights(self):
#         model_dict = {v.name: v for v in self._model_variable}
#         return TransferableWeights(self.sess.run(model_dict))
#
#     def get_gradient_weights(self, batch) -> TransferableWeights:
#         grads = self.sess.run(self._gradient, feed_dict=self._train_feed_dict_fn(batch))
#         return TransferableWeights({v.name: g[0] for g, v in zip(grads, self._gradient_variables)})
#
#     def assign_model_weights(self, weights):
#         trainable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#         self.sess.run([tf.assign(v, weights[v.name]) for v in trainable])
#
#     def apply_gradient_weights(self, weights: TransferableWeights):
#         grads = [weights[v.name] for v in self._gradient_variables]
#         feed_dict = {k[0]: v for k, v in zip(self._grads_holder, grads)}
#         self.sess.run(self._apply_grad_op, feed_dict=feed_dict)
#
#     def save_model(self):
#         print("saving")
#         input_ops = self._predict_inputs
#         output_ops = self._predict_outputs
#         frozen_graph_def = tf.graph_util.convert_variables_to_constants(
#             self.sess,
#             self.sess.graph_def,
#             [v.name.split(":")[0] for v in output_ops.values()]
#         )
#         pb = TFModelParam()
#         pb.model_buf = frozen_graph_def.SerializeToString()
#         for k, v in input_ops.items():
#             self._tensor_info_to_pb(pb.input.add(), k, v)
#
#         for k, v in output_ops.items():
#             self._tensor_info_to_pb(pb.predict.add(), k, v)
#
#         # todo: saving to eggroll
#         with open("model.pb", "wb") as f:
#             f.write(pb.SerializeToString())
#
#     @staticmethod
#     def _tensor_info_to_pb(t: Tensor, name, v: tf.Tensor):
#         t.name = name
#         t.op_name = v.name.split(":")[0]
#         t.d_type = re.findall(r"<dtype: '(\w+?)(?:_ref)?'>", str(v.dtype))[0]
#         if str(v.shape) == "<unknown>":
#             t.shape = "0:"
#         else:
#             num_dim = len(v.shape)
#             dims = ','.join([str(d.value) for d in v.shape])
#             t.shape = f"{num_dim}:{dims}"
#
#     def load_model(self):
#         pass
#
#     def get_batch_accuracy(self, batch):
#         return self._accuracy.eval(session=self.sess, feed_dict=self._eval_feed_dict_fn(batch))
#
#     def train_local_batch(self, batch):
#         return self.sess.run(self._train_op, feed_dict=self._train_feed_dict_fn(batch))
#
#     def train_local(self, data_instances):
#         for batch_iter in range(self.num_batch):
#             batch = data_instances.train.next_batch(self.batch_size)
#             self.train_local_batch(batch)
#             if batch_iter % 100 == 0:
#                 accuracy = self.get_batch_accuracy(batch)
#                 print(f"accuracy={accuracy}")
#
#     def evaluate(self, data_instances):
#         pass
#
#
# class LRModel(Model):
#     def __init__(self, model_shape, initializer, init_param_obj, fit_intercept, gradient_operator,
#                  encrypt_operator, aggregator, updater, optimizer):
#         self.gradient_operator = gradient_operator
#         self.encrypt_operator = encrypt_operator
#         self.fit_intercept = fit_intercept
#         self.aggregator = aggregator
#         self.updater = updater
#         self.optimizer = optimizer
#         self.loss_history = []
#         self.batch_size = 1000
#         w = initializer.init_model(model_shape, init_params=init_param_obj)
#         w = self.encrypt_operator.encrypt_list(w)
#         if fit_intercept:
#             self._coef = w[:-1]
#             self._intercept = w[-1]
#         else:
#             self._coef = w
#             self._intercept = 0
#
#     def get_gradient_weights(self, batch) -> TransferableWeights:
#         raise NotImplementedError("not implemented")
#
#     def get_model_weights(self) -> TransferableWeights:
#         return TransferableWeights(dict(coef=self._coef, intercept=self._intercept))
#
#     def apply_gradient_weights(self, weights: TransferableWeights):
#         raise NotImplementedError("not implemented")
#
#     def assign_model_weights(self, weights: TransferableWeights):
#         self._coef = weights["coef"]
#         self._intercept = weights["intercept"]
#
#     def _update_model(self, gradient):
#         if self.fit_intercept:
#             if self.updater is not None:
#                 self._coef = self.updater.update_coef(self._coef, gradient[:-1])
#             else:
#                 self._coef = self._coef - gradient[:-1]
#             self._intercept -= gradient[-1]
#
#         else:
#             if self.updater is not None:
#                 self._coef = self.updater.update_coef(self._coef, gradient)
#             else:
#                 self._coef = self._coef - gradient
#
#     def _merge_model(self):
#         w = self._coef.copy()
#         if self.fit_intercept:
#             w = np.append(w, self._intercept)
#         return w
#
#     def train_local(self, data_instances):
#         mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)
#         batch_data_generator = mini_batch_obj.mini_batch_data_generator()
#         total_loss = 0
#         batch_num = 0
#
#         for batch_data in batch_data_generator:
#             n = batch_data.count()
#
#             f = functools.partial(self.gradient_operator.compute,
#                                   coef=self._coef,
#                                   intercept=self._intercept,
#                                   fit_intercept=self.fit_intercept)
#             grad_loss = batch_data.mapPartitions(f)
#
#             grad, loss = grad_loss.reduce(self.aggregator.aggregate_grad_loss)
#
#             grad /= n
#             loss /= n
#
#             if self.updater is not None:
#                 loss_norm = self.updater.loss_norm(self._coef)
#                 total_loss += (loss + loss_norm)
#             delta_grad = self.optimizer.apply_gradients(grad)
#
#             self._update_model(delta_grad)
#             batch_num += 1
#
#         total_loss /= batch_num
#         w = self._merge_model()
#         self.loss_history.append(total_loss)
