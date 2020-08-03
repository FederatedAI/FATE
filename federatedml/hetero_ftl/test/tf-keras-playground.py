from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.backend import set_session


def _init_session():
    from tensorflow.python.keras import backend
    sess = backend.get_session()
    tf.get_default_graph()
    set_session(sess)
    return sess


def build_guest_bottom_model():
    model = Sequential()
    model.add(Dense(units=4, input_shape=(2, ), activation='relu', kernel_initializer=keras.initializers.Constant(value=1)))
    # model.add(Dense(units=2, input_shape=(10, ), activation='relu', kernel_initializer='random_uniform'))
    return model


# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
model = guest_bottom_nn_define = build_guest_bottom_model()
optimizer = tf.keras.optimizers.SGD(0.01)
model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

sess = _init_session()

sess.run(tf.global_variables_initializer())

grad_w = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
grad_b = np.array([1, 1, 1, 1])

sess.run(tf.variables_initializer(model.optimizer.variables()))

uninitialized_var_names = [bytes.decode(var) for var in sess.run(tf.report_uninitialized_variables())]
uninitialized_vars = [var for var in tf.global_variables() if var.name.split(':')[0] in uninitialized_var_names]
sess.run(tf.initialize_variables(uninitialized_vars))

op = model.optimizer.apply_gradients([(grad_w, model.trainable_variables[0]), (grad_b, model.trainable_variables[1])])

print(model.get_weights())
sess.run(op)

# out_tensor = model.output * 2
# loss = tf.reduce_sum(model.output)
#
# in_data = np.array([[i, i] for i in range(32)])
#
# # d_loss_d_out_tensor = tf.gradients(loss, model.output)
# #
# # d_loss_d_weight = tf.gradients(loss, model.trainable_variables[0])
# #
# # gradient = tf.gradients(model.output, model.trainable_variables[0])
# #
#


# def copy_a_model(model: Dense):
#
#     dense = Dense(units=model.units, input_shape=model.input_shape,
#                   activation=model.activation, kernel_initializer=model.kernel_initializer)
#     dense.build(input_shape=model.input_shape)
#     dense.kernel = tf.identity(model.kernel)
#     dense.bias = tf.identity(model.bias)
#     dense._trainable_weights.append(dense.kernel)
#     dense._trainable_weights.append(dense.bias)
#     return dense
#
#
# def copy_a_sequential(seq: Sequential):
#     new_seq = Sequential()
#     for l in seq.layers:
#         new_seq.add(copy_a_model(l))
#
#     return new_seq
#
#
# def generate_batch_sequential(seq, batch_size=32):
#
#     seq_list = []
#     for i in range(batch_size):
#         seq_list.append(copy_a_sequential(seq))
#
#     return seq_list
#
#
# def batch_backward_computation(seq_list: List[Sequential], batch_samples, sess: tf.Session,):
#
#     assert len(batch_samples) <= len(seq_list)
#     batch_samples = np.expand_dims(in_data, axis=1)
#     sample_num = len(batch_samples)
#
#     outs = [seq_list[i](batch_samples[i]) for i in range(sample_num)]
#     train_var_len = len(seq_list[0].trainable_variables)
#
#     stacked_outs = tf.stack(outs)
#     train_vars = []
#     for i in range(len(outs)):
#         train_vars.extend(seq_list[i].trainable_variables)
#
#     grads = tf.gradients(stacked_outs, train_vars)
#     computed_gradient = sess.run(grads)
#     return [computed_gradient[i:i+train_var_len] for i in range(0, sample_num, train_var_len)]
#     # return computed_gradient

#
# # d_loss_d_weight_, d_loss_d_out_tensor_ = sess.run([d_loss_d_weight, d_loss_d_out_tensor], feed_dict={model.input: in_data})
# #
# # grad_collect = []
# # for i in range(len(in_data)):
# #     grads_per_sample = sess.run([gradient], feed_dict={model.input: in_data[i:i+1]})
# #     grad_collect.append(grads_per_sample[0][0])
# #
# # per_sample_gradient = np.array(grad_collect)
# # d_loss_d_out_tensor_ = d_loss_d_out_tensor_[0]
# # d_loss_d_out_tensor_ = np.expand_dims(d_loss_d_out_tensor_, axis=1)
# #
# # d_loss_d_weight_2 = np.sum(per_sample_gradient * d_loss_d_out_tensor_, axis=0)
# #
# # print(d_loss_d_weight_)
# # print(d_loss_d_weight_2)
#
# seq_list = generate_batch_sequential(model, batch_size=32)
# sample_grads = batch_backward_computation(seq_list=seq_list, batch_samples=in_data, sess=sess)
# # sum_grad = np.sum(sample_grads, axis=0)
#
# grad = tf.gradients(model.output, model.trainable_variables[1])
# grad2 = sess.run(grad, feed_dict={model.input: in_data})
