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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# from arch.api.eggroll import init
from federatedml.ftl.autoencoder import Autoencoder
from research.beaver_triples_generation.beaver_triple import fill_op_beaver_triple_matrix_shape, create_beaver_triples
from federatedml.ftl.data_util.common_data_util import split_data_combined
from federatedml.ftl.data_util.uci_credit_card_util import load_UCI_Credit_Card_data
from federatedml.ftl.plain_ftl import PlainFTLGuestModel, PlainFTLHostModel
from research.secret_sharing_based_ftl.secure_sharing_ftl import SecureSharingFTLGuestModel, SecureSharingFTLHostModel, \
    LocalSecureSharingFederatedTransferLearning
from federatedml.ftl.test.mock_models import MockFTLModelParam


def create_mul_op_def(num_overlap_samples, num_non_overlap_samples_host, num_non_overlap_samples_guest, guest_input_dim,
                      host_input_dim, hidden_dim):
    # num_samples_host = num_overlap_samples + num_non_overlap_samples_host
    num_samples_guest = num_overlap_samples + num_non_overlap_samples_guest
    ops = []
    mul_op_def = dict()

    #
    # mul operations for host
    #

    op_id = "mul_op_0"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, 1, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_1_for_host"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, host_input_dim, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, host_input_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_1_for_guest"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_samples_guest, guest_input_dim, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_samples_guest, guest_input_dim, hidden_dim)
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
    mul_op_def[op_id]["X_shape"] = (num_non_overlap_samples_guest, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_non_overlap_samples_guest, hidden_dim)
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
    mul_op_def[op_id]["X_shape"] = (num_samples_guest, hidden_dim, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_samples_guest, hidden_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_8"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_samples_guest, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_samples_guest, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    #
    # mul operations for computing loss
    #

    op_id = "mul_op_9"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, 1, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim, 1)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_10"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (1, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (hidden_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_11"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (1, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (hidden_dim, 1)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_12"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    return mul_op_def, ops


def generate_beaver_triples(mul_op_def, num_epoch=1):
    num_batch = 1
    mul_ops = dict()
    for key, val in mul_op_def.items():
        num_batch = fill_op_beaver_triple_matrix_shape(mul_ops,
                                                       op_id=key,
                                                       X_shape=val["X_shape"],
                                                       Y_shape=val["Y_shape"],
                                                       batch_size=val["batch_size"],
                                                       mul_type=val["mul_type"],
                                                       is_constant=val["is_constant"],
                                                       batch_axis=val["batch_axis"])
        print("num_batch", num_batch)

    global_iters = num_batch * num_epoch
    party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=global_iters, num_batch=num_batch)
    return party_a_bt_map, party_b_bt_map, global_iters


# def series_plot(losses, fscores, aucs, vs3):
#     fig = plt.figure(figsize=(20, 40))
#
#     plt.subplot(411)
#     plt.plot(losses)
#     plt.xlabel('epoch')
#     plt.ylabel('values')
#     plt.title("loss")
#     plt.grid(True)
#
#     plt.subplot(412)
#     plt.plot(fscores)
#     plt.xlabel('epoch')
#     plt.ylabel('values')
#     plt.title("fscore")
#     plt.grid(True)
#
#     plt.subplot(413)
#     plt.plot(aucs)
#     plt.xlabel('epoch')
#     plt.ylabel('values')
#     plt.title("auc")
#     plt.grid(True)
#
#     plt.subplot(414)
#     plt.plot(vs3)
#     plt.xlabel('epoch')
#     plt.ylabel('values')
#     plt.title("auc")
#     plt.grid(True)
#
#     plt.show()


if __name__ == '__main__':

    infile = "../../../../examples/data/UCI_Credit_Card.csv"
    X, y = load_UCI_Credit_Card_data(infile=infile, balanced=True)
    X = X[:1000]
    y = y[:1000]
    X_A, y_A, X_B, y_B, overlap_indexes = split_data_combined(X, y,
                                                              overlap_ratio=0.08,
                                                              b_samples_ratio=0.1,
                                                              n_feature_b=16)

    guest_non_overlap_indexes = np.setdiff1d(range(X_A.shape[0]), overlap_indexes)
    host_non_overlap_indexes = np.setdiff1d(range(X_B.shape[0]), overlap_indexes)

    valid_ratio = 0.5
    test_indexes = host_non_overlap_indexes[int(valid_ratio * len(host_non_overlap_indexes)):]
    x_B_test = X_B[test_indexes]
    y_B_test = y_B[test_indexes]

    print("X_A shape", X_A.shape)
    print("y_A shape", y_A.shape)
    print("X_B shape", X_B.shape)
    print("y_B shape", y_A.shape)

    print("overlap_indexes len", len(overlap_indexes))
    print("host_non_overlap_indexes len", len(host_non_overlap_indexes))
    print("guest_non_overlap_indexes len", len(guest_non_overlap_indexes))
    print("test_indexes len", len(test_indexes))

    print("################################ Create Beaver Triples ############################")
    start_time = time.time()
    hidden_dim = 15
    num_epoch = 10
    mul_op_def, ops = create_mul_op_def(num_overlap_samples=len(overlap_indexes),
                                        num_non_overlap_samples_host=len(host_non_overlap_indexes),
                                        num_non_overlap_samples_guest=len(guest_non_overlap_indexes),
                                        guest_input_dim=17,
                                        host_input_dim=16,
                                        hidden_dim=hidden_dim)

    party_a_bt_map, party_b_bt_map, global_iters = generate_beaver_triples(mul_op_def, num_epoch=num_epoch)
    end_time = time.time()
    beaver_triples_generation_time = end_time - start_time
    print("running time:", beaver_triples_generation_time)

    # TODO: save beaver triples

    print("################################ Build Federated Models ############################")

    tf.reset_default_graph()

    autoencoder_guest = Autoencoder(1)
    autoencoder_host = Autoencoder(2)
    autoencoder_guest.build(input_dim=X_A.shape[-1], hidden_dim=hidden_dim, learning_rate=0.01)
    autoencoder_host.build(input_dim=X_B.shape[-1], hidden_dim=hidden_dim, learning_rate=0.01)

    autoencoder_A = Autoencoder(3)
    autoencoder_B = Autoencoder(4)
    autoencoder_A.build(input_dim=X_A.shape[-1], hidden_dim=hidden_dim, learning_rate=0.01)
    autoencoder_B.build(input_dim=X_B.shape[-1], hidden_dim=hidden_dim, learning_rate=0.01)

    mock_model_param = MockFTLModelParam(alpha=50, gamma=0.01, l2_param=0.01)
    guest = SecureSharingFTLGuestModel(autoencoder_guest, mock_model_param)
    host = SecureSharingFTLHostModel(autoencoder_host, mock_model_param)
    guest.set_bt_map(party_a_bt_map)
    host.set_bt_map(party_b_bt_map)

    # plain version of FTL algorithm for testing purpose
    partyA = PlainFTLGuestModel(autoencoder_A, mock_model_param)
    partyB = PlainFTLHostModel(autoencoder_B, mock_model_param)

    federatedLearning = LocalSecureSharingFederatedTransferLearning(guest=guest, host=host)
    federatedLearning.set_party_A(partyA)
    federatedLearning.set_party_B(partyB)

    print("################################ Train Federated Models ############################")
    predict_threshold = 0.55
    start_time = time.time()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        autoencoder_guest.set_session(sess)
        autoencoder_host.set_session(sess)
        autoencoder_A.set_session(sess)
        autoencoder_B.set_session(sess)

        sess.run(init)

        precision, recall, fscore, auc = 0.0, 0.0, 0.0, 0.0
        losses = []
        v1s = []
        v2s = []
        v3s = []
        fscores = []
        aucs = []
        for iter in range(global_iters):
            # self, X_A, X_B, y, overlap_indexes, non_overlap_indexes, global_index
            loss, v1, v2, v3 = federatedLearning.fit(X_A=X_A, X_B=X_B, y=y_A,
                                                     overlap_indexes=overlap_indexes,
                                                     guest_non_overlap_indexes=guest_non_overlap_indexes,
                                                     global_index=iter)
            losses.append(loss)
            v1s.append(v1)
            v2s.append(v2)
            v3s.append(v3)

            if iter % 5 == 0:
                print("iter", iter, "loss", loss)
                y_pred = federatedLearning.predict(x_B_test)
                y_pred_label = []
                pos_count = 0
                neg_count = 0
                # print("y_pred \n", y_pred)
                for _y in y_pred:
                    if _y <= predict_threshold:
                        neg_count += 1
                        y_pred_label.append(-1)
                    else:
                        pos_count += 1
                        y_pred_label.append(1)
                y_pred_label = np.array(y_pred_label)
                print("neg：", neg_count, "pos:", pos_count)
                precision, recall, fscore, _ = precision_recall_fscore_support(y_B_test, y_pred_label, average="weighted")
                fscores.append(fscore)
                print("fscore:", fscore)
                auc = roc_auc_score(y_B_test, y_pred, average="weighted")
                aucs.append(auc)
                # auc = roc_auc_score(y_B_test, y_pred, average="weighted")
                # aucs.append(auc)

        end_time = time.time()
        print("losses:", losses)
        # series_plot(losses, v1s, v2s, v3s)
        # series_plot(losses, fscores, aucs)
        print("precision, recall, fscore, auc", precision, recall, fscore, auc)
        print("running time", end_time - start_time)
        print("beaver_triples_generation_time", beaver_triples_generation_time)

