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

from federatedml.ftl.autoencoder import Autoencoder
from federatedml.ftl.data_util.common_data_util import series_plot, split_data_combined
from federatedml.ftl.data_util.uci_credit_card_util import load_UCI_Credit_Card_data
from federatedml.ftl.plain_ftl import PlainFTLHostModel, PlainFTLGuestModel, LocalPlainFederatedTransferLearning
from federatedml.ftl.test.mock_models import MockFTLModelParam
from federatedml.ftl.data_util.nus_wide_util import get_labeled_data, balance_X_y, get_top_k_labels

if __name__ == '__main__':

    # infile = "../../../../examples/data/UCI_Credit_Card.csv"
    # X, y = load_UCI_Credit_Card_data(infile=infile, balanced=True)
    #
    # num_samples = 5000
    # X = X[:num_samples]
    # y = y[:num_samples]
    # X_A, y_A, X_B, y_B, overlap_indexes = split_data_combined(X, y,
    #                                                           overlap_ratio=0.2,
    #                                                           b_samples_ratio=0.1,
    #                                                           n_feature_b=18)
    #
    # guest_non_overlap_indexes = np.setdiff1d(range(X_A.shape[0]), overlap_indexes)
    # host_non_overlap_indexes = np.setdiff1d(range(X_B.shape[0]), overlap_indexes)
    #
    # valid_ratio = 0.5
    # test_indexes = host_non_overlap_indexes[int(valid_ratio * len(host_non_overlap_indexes)):]
    # X_B_test = X_B[test_indexes]
    # y_B_test = y_B[test_indexes]
    #
    # print("X_A shape", X_A.shape)
    # print("y_A shape", y_A.shape)
    # print("X_B shape", X_B.shape)
    # print("y_B shape", y_B.shape)
    #
    # print("overlap_indexes len", len(overlap_indexes))
    # print("host_non_overlap_indexes len", len(host_non_overlap_indexes))
    # print("guest_non_overlap_indexes len", len(guest_non_overlap_indexes))
    # print("test_indexes len", len(test_indexes))

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
    X_A_test, X_B_test, y_B_test = X_A[num_train:, :], X_B[num_train:, :], y[num_train:, :]
    X_A, X_B, y_A = X_A[:num_train, :], X_B[:num_train, :], y[:num_train, :]

    guest_non_overlap_indexes = np.setdiff1d(range(X_A.shape[0]), overlap_indexes)
    host_non_overlap_indexes = np.setdiff1d(range(X_B.shape[0]), overlap_indexes)

    print("X_A shape", X_A.shape)
    print("X_B shape", X_B.shape)
    print("X_A_test shape", X_A_test.shape)
    print("X_B_test shape", X_B_test.shape)

    print("overlap_indexes len", len(overlap_indexes))
    print("guest_non_overlap_indexes len", len(guest_non_overlap_indexes))
    print("host_non_overlap_indexes len", len(host_non_overlap_indexes))

    print("################################ Build Federated Models ############################")

    tf.reset_default_graph()

    autoencoder_A = Autoencoder(1)
    autoencoder_B = Autoencoder(2)

    hidden_dim = 32
    autoencoder_A.build(X_A.shape[-1], hidden_dim, learning_rate=0.01, repr_l2_param=0.03)
    autoencoder_B.build(X_B.shape[-1], hidden_dim, learning_rate=0.01, repr_l2_param=0.03)

    mock_model_param = MockFTLModelParam(alpha=100, gamma=0.08)
    partyA = PlainFTLGuestModel(autoencoder_A, mock_model_param)
    partyB = PlainFTLHostModel(autoencoder_B, mock_model_param)

    federatedLearning = LocalPlainFederatedTransferLearning(partyA, partyB)

    print("################################ Train Federated Models ############################")
    threshold = 0.50
    start_time = time.time()
    epochs = 50
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        autoencoder_A.set_session(sess)
        autoencoder_B.set_session(sess)

        sess.run(init)

        precision, recall, fscore, auc = 0.0, 0.0, 0.0, 0.0
        losses = []
        fscores = []
        aucs = []
        for ep in range(epochs):
            loss = federatedLearning.fit(X_A=X_A, X_B=X_B, y=y_A,
                                         overlap_indexes=overlap_indexes,
                                         guest_non_overlap_indexes=guest_non_overlap_indexes)
            losses.append(loss)

            if ep % 5 == 0:
                print("> ep", ep, "loss", loss)
                y_pred = federatedLearning.predict(X_B_test)
                y_pred_label = []
                pos_count = 0
                neg_count = 0
                for _y in y_pred:
                    if _y <= threshold:
                        neg_count += 1
                        y_pred_label.append(-1)
                    else:
                        pos_count += 1
                        y_pred_label.append(1)
                y_pred_label = np.array(y_pred_label)
                print("| neg：", neg_count, "pos:", pos_count)
                # print("y_pred shape", y_pred, y_pred.shape)
                # print("y_B_test shape", y_B_test, y_B_test.shape)
                precision, recall, fscore, _ = precision_recall_fscore_support(y_B_test, y_pred_label,
                                                                               average="weighted")
                fscores.append(fscore)
                auc = roc_auc_score(y_B_test, y_pred, average="weighted")
                aucs.append(auc)
                print("| fscore, auc:", fscore, auc)
        end_time = time.time()
        series_plot(losses, fscores, aucs)
        print("precision, recall, fscore, auc", precision, recall, fscore, auc)
        print("running time", end_time - start_time)
