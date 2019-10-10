import unittest

import numpy as np

from arch.api import session
from federatedml.feature.instance import Instance
from federatedml.ftl.test.util import assert_matrix
from research.hetero_dnn_logistic_regression.local_model_proxy import BaseLocalModelUpdateProxy
from research.hetero_dnn_logistic_regression.test.mock_models import MockAutoencoder
from research.hetero_dnn_logistic_regression.test.mock_models import MockFATEFederationClient


def create_instance_table(data, index_list):
    indexed_instances = []
    for idx, row in zip(index_list, data):
        inst = Instance(inst_id=idx, features=row)
        indexed_instances.append((idx, inst))

    dtable = session.parallelize(indexed_instances, include_key=True)
    for r in dtable.collect():
        print(r[0], r[1].features)
    return dtable


def create_shared_gradient_table(gradients, index_list):
    indexed_instances = []
    for idx, grad in zip(index_list, gradients):
        indexed_instances.append((idx, grad))

    dtable = session.parallelize(indexed_instances, include_key=True)
    return dtable


class TestDNNLR(unittest.TestCase):

    def test_local_model_proxy_transform(self):
        print("----test_DNNLR_transform----")

        # create mock data
        X = np.array([[1, 2, 3, 4, 5],
                      [4, 5, 6, 7, 8],
                      [7, 8, 9, 10, 11],
                      [9, 3, 6, 7, 8]])

        index_list = [3, 0, 2, 1]
        instance_table = create_instance_table(X, index_list)

        # create mock local model
        Wh = np.array([[2, 4, 6, 8],
                       [3, 5, 7, 9],
                       [8, 9, 12, 14],
                       [5, 6, 7, 8],
                       [1, 4, 2, 1]])
        bh = np.zeros(X.shape[1])
        local_model = MockAutoencoder(0)
        local_model.build(Wh.shape[0], Wh=Wh, bh=bh)

        # create expected transformed features
        trans_features = np.matmul(X, Wh)

        # create local model proxy to be tested
        proxy = BaseLocalModelUpdateProxy()
        proxy.set_model(local_model)

        # run function
        actual_feature_list = []
        trans_feat_dtable, index_tracking_list = proxy.transform(instance_table)
        trans_feat_dict = dict(trans_feat_dtable.collect())
        for idx in index_tracking_list:
            actual_feature_list.append(trans_feat_dict[idx].features)
        actual_trans_features = np.array(actual_feature_list)

        expected_features = [None] * 4
        for idx, row in zip(index_list, trans_features):
            expected_features[idx] = row
        expected_trans_features = np.array(expected_features)

        # assert results
        print("index_tracking_list", index_tracking_list)
        print("actual_features", actual_trans_features)
        print("expected_trans_features", expected_trans_features)
        assert_matrix(expected_trans_features, actual_trans_features)

    def test_local_model_proxy_update_local_model(self):
        print("----test_DNNLR_update_local_model----")

        X = np.array([[1, 2, 3, 4, 5],
                      [4, 5, 6, 7, 8],
                      [7, 8, 9, 10, 11],
                      [9, 3, 6, 7, 8]])

        index_list = [3, 0, 2, 1]
        instance_table = create_instance_table(X, index_list)

        Wh = np.array([[2, 4, 6, 8],
                       [3, 5, 7, 9],
                       [8, 9, 12, 14],
                       [5, 6, 7, 8],
                       [1, 4, 2, 1]])
        bh = np.zeros(X.shape[1])

        local_model = MockAutoencoder(0)
        local_model.build(Wh.shape[0], Wh=Wh, bh=bh)

        federation_client = MockFATEFederationClient()

        proxy = BaseLocalModelUpdateProxy()
        proxy.set_model(local_model)
        proxy.set_federation_client(federation_client)
        dtable, index_tracking_list = proxy.transform(instance_table)

        coef = np.array([6, 8, 10])
        gradients = np.array([2, 4, 6, 12])

        gradient_table = create_shared_gradient_table(gradients, index_list)

        training_info = {"iteration": 10,
                         "batch_index": 1,
                         "index_tracking_list": index_tracking_list,
                         "is_host": False}

        proxy.update_local_model(gradient_table, instance_table, coef, **training_info)

        gradients = gradients.reshape(len(gradients), 1)
        coef = coef.reshape(1, len(coef))
        back_grad = np.matmul(gradients, coef)

        expected_instances = [None] * 4
        expected_back_grad = [None] * 4
        for idx, g, x in zip(index_list, back_grad, X):
            expected_back_grad[idx] = g
            expected_instances[idx] = x
        expected_back_grad = np.array(expected_back_grad)
        expected_instances = np.array(expected_instances)
        actual_back_grad = local_model.get_back_grad()
        actual_instances = local_model.get_X()

        print("expected_instances:", expected_instances)
        print("actual_instances:", actual_instances)
        print("expected_back_grad", expected_back_grad)
        print("actual_back_grad", actual_back_grad)
        assert_matrix(expected_instances, actual_instances)
        assert_matrix(expected_back_grad, actual_back_grad)


if __name__ == '__main__':
    session.init()
    unittest.main()
