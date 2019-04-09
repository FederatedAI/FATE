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

import time

import numpy as np
import tensorflow as tf

from arch.api.eggroll import init
from federatedml.ftl.autoencoder import Autoencoder
from federatedml.ftl.beaver_triple import fill_beaver_triple_shape, create_beaver_triples
from federatedml.ftl.data_util.common_data_util import series_plot, split_data_combined
from federatedml.ftl.data_util.uci_credit_card_util import load_UCI_Credit_Card_data
from federatedml.ftl.secure_sharing_ftl import SecureSharingFTLGuestModel, SecureSharingFTLHostModel, \
    LocalSecureSharingFederatedTransferLearning
from federatedml.ftl.test.mock_models import MockFTLModelParam


def create_mul_op_def(num_overlap_samples, num_non_overlap_samples, hidden_dim):

    num_samples = num_overlap_samples + num_non_overlap_samples
    ops = []
    mul_op_def = dict()
    op_id = "mul_op_0"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, 1, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_1"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, hidden_dim, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_2"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_3"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, 1, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_4"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_5"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_6"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_7"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_samples, hidden_dim, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_samples, hidden_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_8"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_samples, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_samples, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    return mul_op_def, ops


def generate_beaver_triples(mul_op_def):
    num_batch = 1
    mul_ops = dict()
    for key, val in mul_op_def.items():
        num_batch = fill_beaver_triple_shape(mul_ops,
                                             op_id=key,
                                             X_shape=val["X_shape"],
                                             Y_shape=val["Y_shape"],
                                             batch_size=val["batch_size"],
                                             mul_type=val["mul_type"],
                                             is_constant=val["is_constant"],
                                             batch_axis=val["batch_axis"])
        print("num_batch", num_batch)

    num_epoch = 1
    global_iters = num_batch * num_epoch
    party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=global_iters, num_batch=num_batch)
    return party_a_bt_map, party_b_bt_map


if __name__ == '__main__':

    init()
    infile = "../../../../examples/data/UCI_Credit_Card.csv"
    X, y = load_UCI_Credit_Card_data(infile=infile, balanced=True)

    X = X[:500]
    y = y[:500]

    X_A, y_A, X_B, y_B, overlap_indexes = split_data_combined(X, y,
                                                              overlap_ratio=0.1,
                                                              ab_split_ratio=0.1,
                                                              n_feature_b=23)

    valid_ratio = 0.3
    non_overlap_indexes = np.setdiff1d(range(X_B.shape[0]), overlap_indexes)
    validate_indexes = non_overlap_indexes[:int(valid_ratio * len(non_overlap_indexes))]
    test_indexes = non_overlap_indexes[int(valid_ratio*len(non_overlap_indexes)):]
    x_B_valid = X_B[validate_indexes]
    y_B_valid = y_B[validate_indexes]
    x_B_test = X_B[test_indexes]
    y_B_test = y_B[test_indexes]

    print("X_A shape", X_A.shape)
    print("y_A shape", y_A.shape)
    print("X_B shape", X_B.shape)
    print("y_B shape", y_B.shape)

    print("overlap_indexes len", len(overlap_indexes))
    print("non_overlap_indexes len", len(non_overlap_indexes))
    print("validate_indexes len", len(validate_indexes))
    print("test_indexes len", len(test_indexes))


    print("################################ Create Beaver Triples ############################")

    hidden_dim = 32
    mul_op_def, ops = create_mul_op_def(num_overlap_samples=len(overlap_indexes),
                                        num_non_overlap_samples=len(non_overlap_indexes),
                                        hidden_dim=hidden_dim)
    party_a_bt_map, party_b_bt_map = generate_beaver_triples(mul_op_def)

    print("################################ Build Federated Models ############################")

    tf.reset_default_graph()

    autoencoder_A = Autoencoder(1)
    autoencoder_B = Autoencoder(2)

    autoencoder_A.build(X_A.shape[-1], 32, learning_rate=0.01)
    autoencoder_B.build(X_B.shape[-1], 32, learning_rate=0.01)

    mock_model_param = MockFTLModelParam(gamma=0.01)
    guest = SecureSharingFTLGuestModel(autoencoder_A, mock_model_param)
    host = SecureSharingFTLHostModel(autoencoder_B, mock_model_param)
    guest.set_bt_map(party_a_bt_map)
    host.set_bt_map(party_b_bt_map)

    federatedLearning = LocalSecureSharingFederatedTransferLearning(guest=guest, host=host)

    print("################################ Train Federated Models ############################")
    start_time = time.time()
    epochs = 10
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        autoencoder_A.set_session(sess)
        autoencoder_B.set_session(sess)

        sess.run(init)
        losses = []
        fscores = []
        aucs = []
        for ep in range(epochs):
            loss = federatedLearning.fit(X_A, X_B, y_A, overlap_indexes, non_overlap_indexes)
            losses.append(loss)

            # if ep % 1 == 0:
            #     print("ep", ep, "loss", loss)
            #     y_pred = federatedLearning.fit(x_B_test)
            #     y_pred_label = []
            #     pos_count = 0
            #     neg_count = 0
            #     for _y in y_pred:
            #         if _y <= 0.5:
            #             neg_count += 1
            #             y_pred_label.append(-1)
            #         else:
            #             pos_count += 1
            #             y_pred_label.append(1)
            #     y_pred_label = np.array(y_pred_label)
            #     print("neg：", neg_count, "pos:", pos_count)
            #     precision, recall, fscore, _ = precision_recall_fscore_support(y_B_test, y_pred_label, average="weighted")
            #     fscores.append(fscore)
            #     print("fscore:", fscore)
            #     # auc = roc_auc_score(y_B_test, y_pred, average="weighted")
            #     # aucs.append(auc)

        end_time = time.time()
        series_plot(losses, fscores, aucs)
        print("running time", end_time - start_time)

