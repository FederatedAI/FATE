import numpy as np
import copy
import functools
import itertools
import math
from federatedml.model_base import ModelBase
from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.util import LOGGER
from sklearn.linear_model import LinearRegression
from scipy.special import binom
from federatedml.transfer_variable.transfer_class.shap_transfer_variable import SHAPTransferVariable
from federatedml.param.shap_param import KernelSHAPParam
from federatedml.feature.instance import Instance
from fate_arch.session import computing_session as session
from federatedml.util import consts
from federatedml.util import LOGGER

PRED_VAL = 'pred_val'
BASE_VAL = 'base_val'


class HeteroSHAP(ModelBase):

    def __init__(self):
        super(HeteroSHAP, self).__init__()
        self.model = None
        self.run_local = False
        self.sample_id = None
        self.reference_vec_type = 'all_zeros'
        self.model_param = KernelSHAPParam()
        self.transfer_variable = SHAPTransferVariable()
        self.is_multi_eval = True
        self.schema = None
        self.table_partitions = 4

    def _get_reference_vec_from_background_data(self, data_inst, reference_type='median', ret_format='tensor'):

        if reference_type == 'median':
            statistics = MultivariateStatisticalSummary(data_inst, cols_index=-1)
            reference_data_point = statistics.get_median()
            header = data_inst.schema['header']

            if ret_format == 'tensor':
                ret = []
                for h in header:
                    ret.append(reference_data_point[h])
                return np.array(ret)
            
    def _get_reference_vec(self, data_inst, feat_num):

        if self.reference_vec_type == 'all_zeros':
            reference_vec = np.zeros(feat_num)
        else:
            reference_vec = self._get_reference_vec_from_background_data(data_inst)

        return reference_vec

    def _h_func(self, subset_arr, x, reference_vec):

        """
        map numpy subset vector to data instance
        """

        sample_num = len(subset_arr)
        x_mat = np.repeat([x], sample_num, axis=0)
        reference_vec_mat = np.repeat([reference_vec], sample_num, axis=0)
        x_mat[subset_arr] = reference_vec_mat[subset_arr]
        return x_mat

    def _fitting_shap(self, subset_masks, y, weights, base_val, pred_val):

        if self.is_multi_eval:
            # make sure that class num equals
            assert y.shape[1] == base_val.shape[1]
            assert y.shape[1] == pred_val.shape[1]
            class_num = base_val.shape[1]
            phi = []
            for i in range(class_num):
                sub_y = y[::, i]
                sub_base_val = base_val[::, i]
                sub_pred_val = pred_val[::, i]
                sub_phi = self._fitting_one_dim_shap_values(~subset_masks + 0, sub_y, weights, base_val=sub_base_val,
                                                            pred_val=sub_pred_val)
                phi.append(sub_phi)

        else:
            phi = self._fitting_one_dim_shap_values(~subset_masks + 0, y, weights, base_val=base_val, pred_val=pred_val)

        return phi

    def _fitting_one_dim_shap_values(self, X, y, weights, base_val, pred_val):

        """
        fitting shap values using linear regression model
        X: coliation vectors
        y: pred result of samples mapped from coliation vectors
        weights: kernel weights
        base_val: pred val of reference sample
        pred_val: pred val of example to explain
        """

        LOGGER.debug('fitting coalition vectors')

        # remain last one variable to make sure that all shap values sum to the model output

        linear_model = LinearRegression(fit_intercept=False)
        print('training data shape is {}'.format(X.shape))
        assert X.shape[1] >= 2, 'At least two feature needed when fitting shap values'
        y_label = y - base_val
        y_diff = pred_val - base_val

        last_var = X[::, -1]
        y_label = y_label - y_diff*last_var

        new_X = X[::, :-1] - X[::, -1:]

        print('y label is {}, training x is {}'.format(y_label, new_X))
        linear_model.fit(new_X, y_label, sample_weight=weights)
        phi_partial = list(linear_model.coef_)
        phi_last = (pred_val - base_val) - sum(phi_partial)
        phi_bias = base_val
        phi_partial += list(phi_last)
        phi_partial += list(phi_bias)
        phi = phi_partial
        print('phi {}'.format(phi))
        print('sample sum {}'.format(sum(phi)))

        return phi

    def load_test_model(self, model):
        self.model = model


class HeteroKernelSHAPGuest(HeteroSHAP):

    def __init__(self):
        super(HeteroKernelSHAPGuest, self).__init__()
        self.subset_table = None
        self.subset_set = None
        self.selected_idx_sent = False
        self.class_order = None

    def fit(self, data_inst):

        # initialize
        x = data_inst.take(1)[0][1].features
        self.schema = copy.deepcopy(data_inst.schema)
        self.table_partitions = data_inst.partitions
        if self.run_local:
            phi = self.local_shap(x, data_inst)
            return phi
        else:
            phi = self.federated_shap(x, data_inst)
            return phi

    def _generate_random_sample_ids(self, sample_num):
        return [str(i) for i in range(sample_num)]

    def _make_guest_sample_table(self, x_mat, x, reference_vec):

        assert len(x_mat) == len(self.sample_id), 'sample id len and x_mat len not matched'
        data_list = [(id_, Instance(features=arr)) for id_, arr in zip(self.sample_id, x_mat)]
        data_list.append((PRED_VAL, Instance(features=x)))
        data_list.append((BASE_VAL, Instance(features=reference_vec)))
        guest_table = session.parallelize(data_list, self.table_partitions, include_key=True)
        guest_table.schema = copy.deepcopy(self.schema)
        return guest_table

    def local_shap(self, x, background_data, fate_model=True):

        data_inst = background_data
        header = data_inst.schema['header']
        feat_num = len(header)
        print('feat num is {}'.format(feat_num))
        if feat_num < 1:
            raise ValueError('Too less feature to run LocalKernelSHAP')

        header_idx = [i for i in range(len(header))]
        reference_vec = self._get_reference_vec(data_inst, feat_num)
        subset_masks, sample_weights = self._sample_subsets(feat_num)

        # generate samples: coalition vectors and corresponding predict values
        subset_masks = np.array(subset_masks)
        x_mat = self._h_func(subset_masks, x, reference_vec)

        if fate_model:  # this branch is for running homo fate model
            x_table_list = [(str(i), Instance(features=row)) for i, row in zip(list(range(len(x_mat))), x_mat)]
            x_table_list.append((BASE_VAL, Instance(features=reference_vec)))
            x_table_list.append((PRED_VAL, Instance(features=x)))
            x_table = session.parallelize(x_table_list, partition=self.table_partitions, include_key=True)
            pred_rs = self.model.predict(x_table)
            y, base_val, pred_val = self._reformat_pred_rs(list(pred_rs.collect()))
        else:  # this branch is for running local model
            y = self.model.predict(x_mat)
            base_val = self.model.predict(reference_vec.reshape(1, -1))
            pred_val = self.model.predict(x.reshape(1, -1))

        phi = self._fitting_shap(subset_masks, y, sample_weights, base_val, pred_val)

        return phi

    def _extract_score_list(self, score_list):

        if not self.is_multi_eval:
            return score_list[2]
        else:
            scores = []
            score_dict = score_list[3]
            class_order = []
            for k, v in score_dict.items():
                if self.class_order is None:
                    class_order.append(k)
                scores.append(v)
            if self.class_order is None:
                self.class_order = class_order
            return scores

    def _reformat_pred_rs(self, pred_rs_list):

        base_val = None
        pred_val = None

        y = [None for i in range(len(pred_rs_list) - 2)]
        for id_, score_list in pred_rs_list:
            score = self._extract_score_list(score_list)
            if id_ == PRED_VAL:
                pred_val = np.array([score])
            elif id_ == BASE_VAL:
                base_val = np.array([score])
            else:
                # make sure pred rs in order
                y[int(id_)] = score

        y = np.array(y)
        return y, pred_val, base_val

    def federated_shap(self, x, background_data, host_num=1):

        data_inst = background_data
        header = data_inst.schema['header']
        local_feat_num = len(header)
        print('feat num is {}'.format(local_feat_num))
        if local_feat_num == 0:
            raise ValueError('Too less feature to run LocalKernelSHAP')

        header_idx = [i for i in range(len(header))]
        local_reference_vec = self._get_reference_vec(data_inst, local_feat_num)
        # host_num = len(self.component_properties.host_party_idlist)
        assert host_num >= 1, 'No host found in HeteroKernelSHAP'
        fed_feat_num = host_num
        feat_num = local_feat_num + fed_feat_num
        subset_masks, sample_weights = self._sample_subsets(feat_num)
        subset_masks = np.array(subset_masks)
        local_x_mat = self._h_func(subset_masks[::, 0:local_feat_num], x, local_reference_vec)

        # generate random sample ids
        if self.sample_id is None:
            self.sample_id = self._generate_random_sample_ids(len(subset_masks))
            self.transfer_variable.random_sample_id.remote(self.sample_id, role=consts.HOST, idx=-1)

        if not self.selected_idx_sent:
            # federation part
            for host_idx in range(fed_feat_num):
                selected_idx = subset_masks[::, local_feat_num + host_idx]
                # remote sample_ids and selected index to host
                self.transfer_variable.selected_sample_index.remote(selected_idx, role=consts.HOST, idx=host_idx)
                print('selected idx {}'.format(selected_idx))

        guest_table = self._make_guest_sample_table(local_x_mat, x, local_reference_vec)
        # get predict result of federation samples
        pred_rs_table = self.model.predict(guest_table)
        pred_rs = list(pred_rs_table.collect())
        LOGGER.debug('pred rs is {}'.format(pred_rs))

        y, pred_val, base_val = self._reformat_pred_rs(pred_rs)

        # fitting SHAP value here
        phi = self._fitting_shap(subset_masks, y, sample_weights, base_val, pred_val)
        LOGGER.debug('contrib {}, pred val {}'.format(phi, pred_val))
        return None

    def _compute_subset_budget(self, budget, weights, idx):
        weight = weights[idx] / weights[idx:].sum()
        return budget*weight

    def _get_sample_budget(self, feat_num):
        suggested_val = 2**11 + 2*feat_num
        sample_budget = 2**feat_num - 2
        if sample_budget > suggested_val:
            sample_budget = suggested_val
        return int(sample_budget)

    def _sample_subsets(self, feat_num):

        # sample available subset

        # max sample num
        sample_budget = self._get_sample_budget(feat_num)
        fully_sampled_subset_num = 0
        feat_idx = np.array([i for i in range(feat_num)])

        # initialize
        # size of subset, C^{n}_{m} = C^{m-n}_{m}, simplify subsample by sampling paired subset size at same time
        max_subset_sizes = np.int(np.ceil((feat_num-1)/2))
        max_paired_subset_sizes = np.int(np.floor((feat_num-1)/2))

        # subset weight in SHAP kernel: subset_weight = (M-1) / z(M-z), subsample smaller subsets because
        # their larger weights
        subset_weights = np.array([(feat_num-1)/(i*(feat_num-i)) for i in range(1, max_subset_sizes+1)])
        subset_weights[:max_paired_subset_sizes] *= 2  # paired subset has same weight
        subset_weights /= np.sum(subset_weights)  # normalize

        print('subset weight {}'.format(subset_weights))
        print('sample budget {}'.format(sample_budget))

        subset_masks = []
        sample_weights = []
        for s_idx, s_size in enumerate(range(1, max_subset_sizes+1)):

            fully_sampled_num = binom(feat_num, s_size)  # the sample num of enumerate all subsets of this sizes

            if s_size <= max_paired_subset_sizes:
                fully_sampled_num *= 2

            subset_budget = self._compute_subset_budget(sample_budget, subset_weights, s_idx)
            print('fully sample num {}, budget {}'.format(fully_sampled_num, subset_budget))

            print('judge rs {}'.format(fully_sampled_num / subset_budget))
            if not (subset_budget / fully_sampled_num) >= (1 - 1e-8):  # not able to enumerate all subsets
                print('break')
                break

            print('able to enumerate size {}'.format(s_size))
            fully_sampled_subset_num += 1
            sample_budget -= fully_sampled_num
            w = subset_weights[s_idx] / binom(feat_num, s_size)
            if s_size <= max_paired_subset_sizes:
                w /= 2

            for s in itertools.combinations(feat_idx, s_size):
                subset_mask = np.zeros(feat_num, dtype=bool)
                subset_mask[np.array(s)] = True
                subset_masks.append(~subset_mask)
                sample_weights.append(w)

                if s_size <= max_paired_subset_sizes:
                    subset_masks.append(subset_mask)
                    sample_weights.append(w)

            print('enumerate result length: {}'.format(len(subset_masks)))

        # use rest sample budget
        print(fully_sampled_subset_num == max_subset_sizes)
        print('fully sampled subset num {}, max subset size {}'.format(fully_sampled_subset_num, max_subset_sizes))
        print('sample left {}'.format(sample_budget))
        assert len(sample_weights) == len(subset_masks)
        used_mask = {}
        enumerate_num = len(subset_masks)
        if fully_sampled_subset_num < max_subset_sizes:

            tmp = copy.deepcopy(subset_weights)
            tmp[:max_paired_subset_sizes] /= 2
            remain_weights = tmp[fully_sampled_subset_num:] / tmp[fully_sampled_subset_num:].sum()
            # random sample from rest subset size
            print(len(remain_weights), int(4*sample_budget), remain_weights)
            random_sample_rs = np.random.choice(len(remain_weights), int(4*sample_budget), p=remain_weights)

            for s in random_sample_rs:

                if sample_budget == 0:
                    break

                mask = np.zeros(feat_num, dtype=np.bool)
                s_size = s + fully_sampled_subset_num + 1
                mask[np.random.permutation(feat_num)[:s_size]] = True
                t = tuple(mask)
                if t not in used_mask:
                    used_mask[t] = len(subset_masks)
                    subset_masks.append(mask)
                    sample_weights.append(1)  # add weight
                    sample_budget -= 1
                    if sample_budget == 0:
                        break
                    subset_masks.append(~mask)
                    sample_weights.append(1)  # add complement
                    sample_budget -= 1
                else:
                    sample_weights[used_mask[t]] += 1
                    sample_weights[used_mask[t]+1] += 1

            # re-weight random samples
            remain_weight_sum = subset_weights[fully_sampled_subset_num:].sum()
            sample_weights = np.array(sample_weights)
            sample_weights[enumerate_num:] *= remain_weight_sum / sample_weights[enumerate_num:].sum()
            subset_masks = np.array(subset_masks)

        return subset_masks, sample_weights

    def load_model(self, model_dict):

        from federatedml.ensemble.boosting.hetero.hetero_secureboost_guest import HeteroSecureBoostingTreeGuest
        self.model = HeteroSecureBoostingTreeGuest()
        self.model.load_model(model_dict, model_key='isometric_model')
        self.model.set_flowid(self.flowid)
        self.model.component_properties = copy.deepcopy(self.component_properties)


class HeteroKernelSHAPHost(HeteroSHAP):

    def __init__(self):
        super(HeteroKernelSHAPHost, self).__init__()
        self.sample_id = None
        self.selected_idx = None

    def fit(self, data_inst):

        # initialize
        self.table_partitions = data_inst.partitions
        self.schema = copy.deepcopy(data_inst.schema)
        x = data_inst.take(1)[0][1].features
        self.federated_shap(x)

    def _make_host_sample_table(self, x, reference_data,):

        data_list = []
        for id_, selected in zip(self.sample_id, self.selected_idx):
            if selected:
                data_list.append((id_, Instance(features=x)))
            else:
                data_list.append((id_, Instance(features=reference_data)))
        data_list.append((PRED_VAL, Instance(features=x)))
        data_list.append((BASE_VAL, Instance(features=reference_data)))
        host_table = session.parallelize(data_list, include_key=True, partition=self.table_partitions)
        host_table.schema = copy.deepcopy(self.schema)
        return host_table

    def federated_shap(self, x):

        LOGGER.debug('receiving data')
        if self.sample_id is None:
            self.sample_id = self.transfer_variable.random_sample_id.get(idx=0)
        if self.selected_idx is None:
            self.selected_idx = self.transfer_variable.selected_sample_index.get(idx=0)
        LOGGER.debug('receiving data done')
        local_ref_vec = self._get_reference_vec(None, feat_num=len(x))
        host_table = self._make_host_sample_table(x, local_ref_vec)
        LOGGER.debug('start predict')
        self.model.predict(host_table)
        LOGGER.debug('predict done')

    def load_model(self, model_dict):

        from federatedml.ensemble.boosting.hetero.hetero_secureboost_host import HeteroSecureBoostingTreeHost
        self.model = HeteroSecureBoostingTreeHost()
        self.model.load_model(model_dict, model_key='isometric_model')
        self.model.set_flowid(self.flowid)
        self.model.component_properties = copy.deepcopy(self.component_properties)


if __name__ == '__main__':

    from federatedml.model_interpret.test.local_test import get_breast_guest, session, get_vehicle_guest
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import scipy.special
    import numpy as np
    import itertools


    def powerset(iterable):
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


    def shapley_kernel(M, s):
        if s == 0 or s == M:
            return 10000
        return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))


    def kernel_shap(f, x, reference, M):

        X = np.zeros((2 ** M, M + 1))
        X[:, -1] = 1
        weights = np.zeros(2 ** M)
        V = np.zeros((2 ** M, M))
        for i in range(2 ** M):
            V[i, :] = reference

        ws = {}
        for i, s in enumerate(powerset(range(M))):
            s = list(s)
            V[i, s] = x[s]
            X[i, s] = 1
            ws[len(s)] = ws.get(len(s), 0) + shapley_kernel(M, len(s))
            weights[i] = shapley_kernel(M, len(s))
        y = f(V)
        tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
        return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))


    df = pd.read_csv('../../../examples/data/breast_hetero_guest.csv')
    model = LogisticRegression()
    train = df.drop(columns=['y', 'id']).values
    model.fit(train, df['y'])

    df_1 = pd.read_csv('../../../examples/data/vehicle_scale_homo_guest.csv')
    model_1 = RandomForestClassifier(random_state=114)
    train_1 = df_1.drop(columns=['y', 'id']).values
    model_1.fit(train_1, df_1['y'])

    class ModelAdapter():

        def __init__(self, model):
            self.model = model

        def predict(self, X):
            return self.model.predict_proba(X)[:, 0]


    class MultiModelAdapter(ModelAdapter):

        def __init__(self, model):
            super(MultiModelAdapter, self).__init__(model)

        def predict(self, X):
            return self.model.predict_proba(X)

    guest_table = get_breast_guest()
    guest_table_2 = get_vehicle_guest()

    model_1_ = MultiModelAdapter(model_1)
    shap_2 = HeteroKernelSHAPGuest()
    shap_2.model = model_1_
    shap_2.is_multi_eval = True
    shap_rs_3 = shap_2.local_shap(train_1[0], guest_table_2, fate_model=False)
    x_pred_2 = model_1_.predict(train_1[0].reshape(1, -1))


    # x = train[0]
    # x_predict = model.predict_proba(x.reshape(1, -1))
    #
    # shap = HeteroKernelSHAPGuest()
    # subset_masks, sample_weights = shap._sample_subsets(x.shape[0])
    #
    # model_ = ModelAdapter(model)
    # shap.load_test_model(model_)
    # # shap_rs = shap.local_shap(x, guest_table)
    #
    # shap_rs_2 = shap.federated_shap(x, guest_table, 1)
