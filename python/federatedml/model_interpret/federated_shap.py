import numpy as np
import copy
import functools
import itertools
import math
from federatedml.model_base import ModelBase
from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.util import LOGGER
from sklearn.linear_model import LinearRegression
from fate_arch.session import computing_session as session


class HeteroSHAP(ModelBase):

    def __init__(self):
        super(HeteroSHAP, self).__init__()
        self.model = None
        self.run_local = True

    def _shapley_kernel(self, M, z):

        """
        Shapely kernel defined in paper
        """

        if z == 0 or z == M:
            return 10000

        M_choose_s = math.factorial(M) / (math.factorial(z) * math.factorial(M-z))
        weight = (M-1)/(M_choose_s*z*(M-z))
        return weight

    def _get_reference_data_point(self, data_inst, reference_type='median', ret_format='tensor'):

        if reference_type == 'median':
            statistics = MultivariateStatisticalSummary(data_inst, cols_index=-1)
            reference_data_point = statistics.get_median()
            header = data_inst.schema['header']

            if ret_format == 'tensor':
                ret = []
                for h in header:
                    ret.append(reference_data_point[h])
                return np.array(ret)

    # subset generating
    def _get_all_subset(self, feats):
        all_subsets = itertools.chain.from_iterable(itertools.combinations(feats, r) for r in range(len(feats) + 1))
        return all_subsets

    def _sample_subsets(self):
        pass

    def _h_func(self, subset, x, reference_vec):
        """
        map coalition to input data instance
        """
        copied_reference_vec = np.array(reference_vec)  # copy reference vector
        indices = list(subset)
        copied_reference_vec[indices] = x[indices]
        return reference_vec

    def _local_generate_coalition_vectors(self, feat_num, subsets):

        vectors = np.zeros((len(subsets), feat_num))
        kernel_weights = np.zeros(len(subsets))
        for i, s in enumerate(subsets):
            vectors[i, list(s)] = 1
            kernel_weights[i] = self._shapley_kernel(feat_num, len(s))
        return vectors, kernel_weights

    def _local_predict_coalition_y(self, subsets, x, reference_vec):
        to_predict = []
        for s in subsets:
            to_predict.append(self._h_func(s, x, reference_vec))
        predicted = self.model.predict(to_predict)
        return predicted

    def _fitting_shap_values(self, X, y, weights, use_sklearn=True):

        LOGGER.debug('fitting coalition vectors')
        if use_sklearn:
            linear_model = LinearRegression(fit_intercept=True)
            linear_model.fit(X=X, y=y, sample_weight=weights)
            phi_ = linear_model.coef_
            phi_bias = linear_model.intercept_
            phi = list(phi_)
            phi.append(phi_bias)
        else:
            # extend bias term
            bias = np.zeros((len(X), 1)) + 1
            X = np.concatenate([X, bias], axis=1)
            # result of linear regression: theta = (X^T X)^-1 X^T Y
            tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
            phi = np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))

        return phi

    def load_model(self, model):
        self.model = model


class HeteroBruteSHAPGuest(HeteroSHAP):

    def __init__(self):
        super(HeteroBruteSHAPGuest, self).__init__()
        self.subset_table = None

    def fit(self, data_inst):

        # initialize
        x = data_inst.take(1)[0][1].features
        if self.run_local:
            phi = self.local_shap(x, data_inst)
            return phi

    def _federated_predict_coliation_y(self, subset_table, x, reference_vec):

        map_h_func = functools.partial(self._h_func, x=x, reference_vec=reference_vec)
        to_predict_data = subset_table.mapValues(map_h_func)
        predict_rs = self.model.predict(to_predict_data)
        return predict_rs

    def local_shap(self, x, background_data):

        data_inst = background_data
        reference_vec = self._get_reference_data_point(data_inst)
        header = data_inst.schema['header']
        header_idx = [i for i in range(len(header))]
        subsets = list(shap._get_all_subset(header_idx))

        # generate samples: coalition vectors and corresponding predict values
        coalition_vectors, weights = shap._local_generate_coalition_vectors(len(header), subsets)
        y = self._local_predict_coalition_y(subsets, x, reference_vec)
        phi = self._fitting_shap_values(coalition_vectors, y, weights, use_sklearn=True)

        return phi

    def federated_shap(self, x, background_data):

        data_inst = background_data
        reference_vec = self._get_reference_data_point(data_inst)
        header = copy.deepcopy(data_inst.schema['header'])
        header.append('fed_feat')
        header_idx = [i for i in range(len(header))]
        subsets = list(shap._get_all_subset(header_idx))
        self.subset_table = session.parallelize(subsets, partition=data_inst.partitions, include_key=False)

        return self.subset_table


class HeteroBruteSHAPHost(HeteroSHAP):

    def __init__(self):
        pass

    def federated_shap(self, x, background_data, fit_type='sklearn'):

        data_inst = background_data
        reference_vec = self._get_reference_data_point(data_inst)


class HeteroTreeSHAP(ModelBase):

    def __init__(self):
        pass


class HeteroTreeSHAPGuest(ModelBase):

    def __init__(self):
        pass


class HeteroTreeSHAPHost(ModelBase):

    def __init__(self):
        pass


if __name__ == '__main__':

    from federatedml.model_interpret.test.local_test import get_breast_guest, session
    from sklearn.linear_model import LogisticRegression
    import pandas as pd

    df = pd.read_csv('../../../examples/data/breast_hetero_guest.csv')

    model = LogisticRegression()
    model.fit(df.drop(columns=['y', 'id']), df['y'])

    class ModelAdapter():

        def __init__(self, model):
            self.model = model

        def predict(self, X):
            return self.model.predict_proba(X)[:, -1]

    guest_table = get_breast_guest()

    x = guest_table.take(1)[0][1].features
    x_predict = model.predict_proba(x.reshape(1, -1))

    shap = HeteroBruteSHAPGuest()
    # shap.load_model(ModelAdapter(model))
    # shap_values = shap.fit(guest_table)
    rs = shap.federated_shap(x, guest_table)