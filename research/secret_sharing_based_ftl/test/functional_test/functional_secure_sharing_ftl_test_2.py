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

import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# from arch.api.eggroll import init
from federatedml.ftl.autoencoder import Autoencoder
from federatedml.ftl.data_util.common_data_util import series_plot
from federatedml.ftl.data_util.nus_wide_util import get_labeled_data, balance_X_y, get_top_k_labels
from federatedml.ftl.plain_ftl import PlainFTLGuestModel, PlainFTLHostModel
from federatedml.ftl.test.mock_models import MockFTLModelParam
from research.beaver_triples_generation.beaver_triple import fill_beaver_triple_matrix_shape, create_beaver_triples
from research.secret_sharing_based_ftl.secure_sharing_ftl import SecureSharingFTLGuestModel, SecureSharingFTLHostModel, \
    LocalSecureSharingFederatedTransferLearning


def create_mul_op_def(num_overlap_samples, num_non_overlap_samples_guest, guest_input_dim, host_input_dim, hidden_dim):
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


# def fill_beaver_triple_matrix_shape(mul_op_def, num_epoch=1):
#     num_batch = 1
#     mul_ops = dict()
#     for op_id, attr in mul_op_def.items():
#         num_batch = fill_op_beaver_triple_matrix_shape(mul_ops,
#                                                        op_id=op_id,
#                                                        X_shape=attr["X_shape"],
#                                                        Y_shape=attr["Y_shape"],
#                                                        batch_size=attr["batch_size"],
#                                                        mul_type=attr["mul_type"],
#                                                        is_constant=attr["is_constant"],
#                                                        batch_axis=attr["batch_axis"])
#         print("num_batch", num_batch)
#     global_iters = num_batch * num_epoch
#     return mul_ops, global_iters, num_batch


def generate_beaver_triples(mul_op_def, num_epoch=1):
    mul_ops, global_iters, num_batch = fill_beaver_triple_matrix_shape(mul_op_def, num_epoch)
    party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=global_iters, num_batch=num_batch)
    return party_a_bt_map, party_b_bt_map, global_iters

# def generate_beaver_triples(mul_op_def, num_epoch=1):
#     num_batch = 1
#     mul_ops = dict()
#     for op_id, attr in mul_op_def.items():
#         num_batch = fill_op_beaver_triple_matrix_shape(mul_ops,
#                                                        op_id=op_id,
#                                                        X_shape=attr["X_shape"],
#                                                        Y_shape=attr["Y_shape"],
#                                                        batch_size=attr["batch_size"],
#                                                        mul_type=attr["mul_type"],
#                                                        is_constant=attr["is_constant"],
#                                                        batch_axis=attr["batch_axis"])
#         print("num_batch", num_batch)
#
#     global_iters = num_batch * num_epoch
#     party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=global_iters, num_batch=num_batch)
#     return party_a_bt_map, party_b_bt_map, global_iters


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

    # infile = "../../../../examples/data/UCI_Credit_Card.csv"
    # X, y = load_UCI_Credit_Card_data(infile=infile, balanced=True)
    # X = X[:2000]
    # y = y[:2000]
    # X_A, y_A, X_B, y_B, overlap_indexes = split_data_combined(X, y,
    #                                                           overlap_ratio=0.2,
    #                                                           b_samples_ratio=0.1,
    #                                                           n_feature_b=16)

    file_dir = "/data/app/fate/yankang/"

    sel = ["person"]
    all_labels = get_top_k_labels(file_dir, top_k=81)
    all_labels.remove("person")

    sel = sel + all_labels
    print("sel:", sel)
    X_A, X_B, y = get_labeled_data(data_dir=file_dir, selected_label=sel, n_samples=5000)
    print("X_A shape:", X_A.shape)
    print("X_B shape:", X_B.shape)
    print("y shape", y.shape)

    # main_label = 'water'
    # num_samples = 5000
    # num_categories = 81
    # all_labels = get_top_k_labels(file_dir, top_k=num_categories)
    # all_labels.remove(main_label)
    # sel = main_label + all_labels
    #
    # model_name_prefix = main_label + "_" + str(num_categories) + "_" + str(num_samples)
    # print("model_name_prefix:", model_name_prefix)
    #
    # rel_model_dir = "models"
    # model_full_path_dir = file_dir + rel_model_dir
    # if not os.path.exists(model_full_path_dir):
    #     try:
    #         os.mkdir(model_full_path_dir)
    #     except OSError:
    #         print("Creation of the directory {0} failed".format(model_full_path_dir))
    #         raise OSError("Creation of the directory {0} failed".format(model_full_path_dir))
    #     else:
    #         print("Successfully created the directory {0}".format(model_full_path_dir))
    # else:
    #     print("directory {0} already exists".format(model_full_path_dir))
    #
    # model_full_name_prefix = model_full_path_dir + "/" + model_name_prefix
    # model_full_name_X_A = model_full_name_prefix + "_" + "X_A"
    # model_full_name_X_B = model_full_name_prefix + "_" + "X_B"
    # model_full_name_y = model_full_name_prefix + "_" + "y"
    #
    # print("model_name_X_A", model_full_name_X_A)
    # print("model_name_X_B", model_full_name_X_B)
    # print("model_name_y", model_full_name_y)
    #
    # if not os.path.exists(model_full_name_X_A):
    #     X_A, X_B, y = get_labeled_data(data_dir=file_dir, selected_label=sel, n_samples=num_samples)
    #     print("original X_A shape:", X_A.shape)
    #     print("original X_B shape:", X_B.shape)
    #     print("original y shape", y.shape)
    #     np.save(model_full_name_X_A, X_A)
    #     np.save(model_full_name_X_B, X_B)
    #     np.save(model_full_name_y, y)
    # else:
    #     X_A = np.load(model_full_name_X_A)
    #     X_B = np.load(model_full_name_X_B)
    #     y = np.load(model_full_name_y)

    y_ = []
    pos_count = 0
    neg_count = 0
    for i in range(y.shape[0]):
        if y[i, 0] == 1:
            y_.append(1)
            pos_count += 1
        else:
            y_.append(-1)
            neg_count += 1

    y_ = np.array(y_)
    X_A, X_B, y = balance_X_y(X_A, X_B, y_)

    y = np.expand_dims(y, axis=1)
    print("X_A shape:", X_A.shape)
    print("X_B shape:", X_B.shape)
    print("y shape:", y.shape)
    # print("y:", y)

    overlap_ratio = 0.4
    data_size = X_A.shape[0]
    overlap_size = int(data_size * overlap_ratio)
    overlap_indexes = np.array(range(overlap_size))

    num_train = int(0.8 * data_size)

    print("num_train:", num_train)
    X_A_test, X_B_test, y_test = X_A[num_train:, :], X_B[num_train:, :], y[num_train:, :]
    X_A, X_B, y = X_A[:num_train, :], X_B[:num_train, :], y[:num_train, :]

    guest_non_overlap_indexes = np.setdiff1d(range(X_A.shape[0]), overlap_indexes)
    host_non_overlap_indexes = np.setdiff1d(range(X_B.shape[0]), overlap_indexes)

    print("X_A shape", X_A.shape)
    print("X_B shape", X_B.shape)
    print("X_A_test shape", X_A_test.shape)
    print("X_B_test shape", X_B_test.shape)

    print("overlap_indexes len", len(overlap_indexes))
    print("guest_non_overlap_indexes len", len(guest_non_overlap_indexes))
    print("host_non_overlap_indexes len", len(host_non_overlap_indexes))

    print("################################ Create Beaver Triples ############################")
    start_time = time.time()
    hidden_dim = 48
    num_epoch = 1
    mul_op_def, ops = create_mul_op_def(num_overlap_samples=len(overlap_indexes),
                                        num_non_overlap_samples_guest=len(guest_non_overlap_indexes),
                                        guest_input_dim=634,
                                        host_input_dim=1000,
                                        hidden_dim=hidden_dim)

    party_a_bt_map, party_b_bt_map, global_iters = generate_beaver_triples(mul_op_def, num_epoch=num_epoch)
    end_time = time.time()
    beaver_triples_generation_time = end_time - start_time
    print("running time:", beaver_triples_generation_time)

    # TODO: save beaver triples
    global_iters = 20
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
            loss, v1, v2, v3 = federatedLearning.fit(X_A=X_A, X_B=X_B, y=y,
                                                     overlap_indexes=overlap_indexes,
                                                     guest_non_overlap_indexes=guest_non_overlap_indexes,
                                                     global_index=iter)
            losses.append(loss)
            v1s.append(v1)
            v2s.append(v2)
            v3s.append(v3)

            if iter % 5 == 0:
                print("iter", iter, "loss", loss)
                y_pred = federatedLearning.predict(X_B_test)
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
                precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred_label)
                fscores.append(fscore)
                print("fscore:", fscore)
                auc = roc_auc_score(y_test, y_pred, average="weighted")
                aucs.append(auc)
                # auc = roc_auc_score(y_B_test, y_pred, average="weighted")
                # aucs.append(auc)

        end_time = time.time()
        print("losses:", losses)
        print("y_test_pos:", np.sum(y_test[y_test == 1]))
        print("y_test_neg:", np.sum(y_test[y_test == -1]))
        # series_plot(losses, v1s, v2s, v3s)
        series_plot(losses, fscores, aucs)
        print("precision, recall, fscore, auc", precision, recall, fscore, auc)
        print("running time", end_time - start_time)
        print("beaver_triples_generation_time", beaver_triples_generation_time)

