import copy
import shap
import numpy as np
import itertools
from scipy.special import binom
from federatedml.util import consts
from federatedml.util import LOGGER
from sklearn.linear_model import LinearRegression
from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.model_interpret.explainer.explainer_base import Explainer, data_inst_table_to_arr
from federatedml.transfer_variable.transfer_class.shap_transfer_variable import SHAPTransferVariable
from federatedml.feature.instance import Instance
from federatedml.param.shap_param import SHAPParam
from fate_arch.session import computing_session as session


PRED_VAL = 'pred_val'
BASE_VAL = 'base_val'


def handle_result(data_arr, shap_result, expected_val):

    # multi classification case
    if type(expected_val) == np.ndarray or type(expected_val) == list:
        rs = []
        for sub_shap_rs, val in zip(shap_result, expected_val):
            expected_val_col = np.zeros((data_arr.shape[0], 1)) + val
            explain_rs = np.concatenate([sub_shap_rs, expected_val_col], axis=1)
            rs.append(explain_rs)
        return rs
    else:
        expected_val_col = np.zeros((data_arr.shape[0], 1)) + expected_val
        explain_rs = np.concatenate([shap_result, expected_val_col], axis=1)
        return explain_rs


class KernelSHAP(Explainer):

    def __init__(self, role, flow_id):
        super(KernelSHAP, self).__init__(role, flow_id)
        
        self.model = None
        self.sample_id = None
        self.reference_vec_type = consts.ZEROS
        self.model_param = SHAPParam()
        self.transfer_variable = SHAPTransferVariable()
        self.transfer_variable.set_flowid(self.flow_id)
        self.is_multi_eval = False
        self.schema = None
        self.table_partitions = 4
        self.component_properties = None

    def init_model(self, model):
        self.model = model

    def set_component_properties(self, component_properties):
        self.component_properties = component_properties

    def set_reference_type(self, ref_type):
        assert ref_type in [consts.MEDIAN, consts.AVERAGE, consts.ZEROS]
        self.reference_vec_type = ref_type


class HomoKernelSHAP(KernelSHAP):

    def __init__(self, role, flow_id):
        super(HomoKernelSHAP, self).__init__(role, flow_id)

    def explain(self, data_inst, n=500):

        ids, header, data_arr = data_inst_table_to_arr(data_inst, n)
        LOGGER.debug('homo kernel shap to explain is {}'.format(data_arr))
        LOGGER.debug('homo kernel shap to explain shape is {}'.format(data_arr.shape))

        ref_vec = None
        if self.reference_vec_type == consts.ZEROS:
            ref_vec = np.zeros((1, data_arr.shape[1]))
        elif self.reference_vec_type == consts.AVERAGE:
            ref_vec = np.array([data_arr.mean(axis=0)])
        elif self.reference_vec_type == consts.MEDIAN:
            ref_vec = np.array([np.median(axis=0)])

        LOGGER.debug('ref vec is {}'.format(ref_vec))
        LOGGER.debug('pred rs is {}'.format(self.model.predict(data_arr)))
        shap_kernel = shap.KernelExplainer(self.model.predict, ref_vec)
        LOGGER.debug('kernel shap on running')
        explain_rs = shap_kernel.shap_values(data_arr)
        explain_rs = handle_result(data_arr, explain_rs, shap_kernel.expected_value)
        LOGGER.debug('homo kernel shap explain rs is {}'.format(explain_rs))


class HeteroKernelSHAP(KernelSHAP):
    
    def __init__(self, role, flow_id):
        super(HeteroKernelSHAP, self).__init__(role, flow_id)
        self.host_mask_sent = False
        self.class_order = None

        # cache
        self.subset_mask = None
        self.subset_weights = None
        self.ref_vec_mat_cache = None

        # full explain not support linear models
        self.full_explain = False
        self.host_side_feat_list = None
        self.host_side_feat_num = None  # this var for full explain
        self.sync_host_feat_num = True
        self.host_mask = None

    def set_full_explain(self):
        self.full_explain = True
        
    def _generate_subset_sample_ids(self, sample_num):
        return [str(i) for i in range(sample_num)]

    def _get_reference_vec_from_background_data(self, data_inst, reference_type=consts.MEDIAN):

        reference_data_point = None
        header = data_inst.schema['header']
        if reference_type == consts.MEDIAN:
            statistics = MultivariateStatisticalSummary(data_inst, cols_index=-1)
            reference_data_point = statistics.get_median()

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
        if self.ref_vec_mat_cache is None:
            self.ref_vec_mat_cache = np.repeat([reference_vec], sample_num, axis=0)
        x_mat[subset_arr] = self.ref_vec_mat_cache[subset_arr]
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
        X: coalition vectors
        y: predict result of samples mapped from coalition vectors
        weights: kernel weights
        base_val: predict val of reference sample
        pred_val: predict val of example to explain
        """

        LOGGER.debug('fitting coalition vectors')

        # remain last one variable to make sure that all shap values sum to the model output

        linear_model = LinearRegression(fit_intercept=False)
        LOGGER.debug('training data shape is {}'.format(X.shape))
        assert X.shape[1] >= 2, 'At least two feature needed when fitting shap values'
        y_label = y - base_val
        y_diff = pred_val - base_val

        last_var = X[::, -1]
        y_label = y_label - y_diff * last_var
        new_X = X[::, :-1] - X[::, -1:]

        LOGGER.debug('y label is {}, training x is {}'.format(y_label, new_X))
        linear_model.fit(new_X, y_label, sample_weight=weights)
        phi_partial = list(linear_model.coef_)
        phi_last = (pred_val - base_val) - sum(phi_partial)
        phi_bias = base_val
        phi_partial += list(phi_last)
        phi_partial += list(phi_bias)
        phi = phi_partial
        LOGGER.debug('phi {}'.format(phi))
        LOGGER.debug('sample sum {}'.format(sum(phi)))

        return phi

    def _compute_subset_budget(self, budget, weights, idx):
        weight = weights[idx] / weights[idx:].sum()
        return budget * weight

    def _get_sample_budget(self, feat_num):
        # give a suggested subsample num
        # this suggested_val is copied from KernelExplainer in SHAP library
        suggested_val = 2 ** 11 + 2 * feat_num
        sample_budget = 2 ** feat_num - 2
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
        # size of subset, C^{n}_{m} = C^{m-n}_{m}, simplify subsample by sampling paired subset size at the same time
        max_subset_sizes = np.int(np.ceil((feat_num - 1) / 2))
        max_paired_subset_sizes = np.int(np.floor((feat_num - 1) / 2))

        # subset weight in SHAP kernel: subset_weight = (M-1) / z(M-z), prefer smaller subsets because
        # they have larger weights
        subset_weights = np.array([(feat_num - 1) / (i * (feat_num - i)) for i in range(1, max_subset_sizes + 1)])
        subset_weights[:max_paired_subset_sizes] *= 2  # paired subset has same weight
        subset_weights /= np.sum(subset_weights)  # normalize

        subset_masks = []
        sample_weights = []
        for s_idx, s_size in enumerate(range(1, max_subset_sizes + 1)):

            fully_sampled_num = binom(feat_num, s_size)  # the sample num of enumerate all subsets of this sizes

            if s_size <= max_paired_subset_sizes:
                fully_sampled_num *= 2

            subset_budget = self._compute_subset_budget(sample_budget, subset_weights, s_idx)
            LOGGER.debug('fully sample num {}, budget {}'.format(fully_sampled_num, subset_budget))

            if not (subset_budget / fully_sampled_num) >= (1 - 1e-8):  # not able to enumerate all subsets
                LOGGER.debug('break')
                break

            LOGGER.debug('able to enumerate size {}'.format(s_size))
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

        # use up rest sample budget
        LOGGER.debug(fully_sampled_subset_num == max_subset_sizes)
        LOGGER.debug(
            'fully sampled subset num {}, max subset size {}'.format(fully_sampled_subset_num, max_subset_sizes))
        LOGGER.debug('sample left {}'.format(sample_budget))
        assert len(sample_weights) == len(subset_masks)
        used_mask = {}
        enumerate_num = len(subset_masks)
        if fully_sampled_subset_num < max_subset_sizes:

            tmp = copy.deepcopy(subset_weights)
            tmp[:max_paired_subset_sizes] /= 2
            # this is a probability computed from remaining weights
            remain_weights = tmp[fully_sampled_subset_num:] / tmp[fully_sampled_subset_num:].sum()
            # random sample from rest subset size
            random_sample_rs = np.random.choice(len(remain_weights), int(4 * sample_budget), p=remain_weights)

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
                    sample_weights[used_mask[t] + 1] += 1

            # re-weight random samples
            remain_weight_sum = subset_weights[fully_sampled_subset_num:].sum()
            sample_weights = np.array(sample_weights)
            sample_weights[enumerate_num:] *= remain_weight_sum / sample_weights[enumerate_num:].sum()
            subset_masks = np.array(subset_masks)

        return subset_masks, sample_weights

    def _make_sample_table(self, x_mat, x, reference_vec):

        assert len(x_mat) == len(self.sample_id), 'sample id len and x_mat len not matched'
        data_list = [(id_, Instance(features=arr)) for id_, arr in zip(self.sample_id, x_mat)]
        data_list.append((PRED_VAL, Instance(features=x)))
        data_list.append((BASE_VAL, Instance(features=reference_vec)))
        guest_table = session.parallelize(data_list, self.table_partitions, include_key=True)
        guest_table.schema = copy.deepcopy(self.schema)
        return guest_table

    def _make_non_full_explain_sample_table(self, x, reference_data):

        data_list = []
        for id_, selected in zip(self.sample_id, self.host_mask):
            if selected:
                data_list.append((id_, Instance(features=x)))
            else:
                data_list.append((id_, Instance(features=reference_data)))
        data_list.append((PRED_VAL, Instance(features=x)))
        data_list.append((BASE_VAL, Instance(features=reference_data)))
        host_table = session.parallelize(data_list, include_key=True, partition=self.table_partitions)
        host_table.schema = copy.deepcopy(self.schema)
        return host_table
    
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
    
    def guest_explain_row(self, x, background_data, host_num=1):

        data_inst = background_data
        header = data_inst.schema['header']
        guest_local_feat_num = len(header)
        LOGGER.debug('feat num is {}'.format(guest_local_feat_num))
        if guest_local_feat_num == 0:
            raise ValueError('guest should offer at least 1 feature')

        local_reference_vec = self._get_reference_vec(data_inst, guest_local_feat_num)

        assert host_num >= 1, 'No host found in HeteroKernelSHAP'
        if self.full_explain:
            host_side_feat_num = self.host_side_feat_num
            feat_num = guest_local_feat_num + host_side_feat_num
        else:
            host_side_feat_num = host_num
            feat_num = guest_local_feat_num + host_side_feat_num

        # subset sampling
        if self.subset_mask is None and self.subset_weights is None:
            # conduct subset sampling
            subset_masks, sample_weights = self._sample_subsets(feat_num)
            subset_masks = np.array(subset_masks)
            self.subset_mask, self.subset_weights = subset_masks, sample_weights
        else:
            # reuse
            subset_masks, sample_weights = self.subset_mask, self.subset_weights

        LOGGER.debug('subset is {} shape is {}'.format(subset_masks, subset_masks.shape))
        # generate data to predict using mask
        local_x_mat = self._h_func(subset_masks[::, 0:guest_local_feat_num], x, local_reference_vec)

        # generate random sample ids
        if self.sample_id is None:
            self.sample_id = self._generate_subset_sample_ids(len(subset_masks))
            self.transfer_variable.random_sample_id.remote(self.sample_id, role=consts.HOST, idx=-1)

        if not self.host_mask_sent:
            # federation part
            for host_idx in range(host_num):
                offset = 0
                start = guest_local_feat_num
                if not self.full_explain:
                    # fed-feature mask, only one column
                    offset = host_idx
                    to_send = subset_masks[::, guest_local_feat_num + offset]
                    # remote sample_ids and selected index to host
                    self.transfer_variable.selected_sample_index.remote(to_send, role=consts.HOST, idx=host_idx)
                else:
                    # host side feature mask, column num corresponds to host feature num
                    offset += self.host_side_feat_list[host_idx]
                    to_send = subset_masks[::, start: start+offset]
                    start += offset
                    self.transfer_variable.selected_sample_index.remote(to_send, role=consts.HOST, idx=host_idx)

                LOGGER.debug('sending mask to host {}'.format(to_send))

        guest_table = self._make_sample_table(local_x_mat, x, local_reference_vec)

        # get predict result of federation models
        pred_rs_table = self.model.predict(guest_table)
        pred_collect = list(pred_rs_table.collect())
        pred_rs = [(k, v.features) for k, v in pred_collect]
        # LOGGER.debug('pred rs is {}'.format(pred_rs))

        y, pred_val, base_val = self._reformat_pred_rs(pred_rs)

        # fitting SHAP value here
        phi = self._fitting_shap(subset_masks, y, sample_weights, base_val, pred_val)
        LOGGER.debug('contrib {}, pred val {}'.format(phi, pred_val))
        return phi

    def host_explain_row(self, x, data_inst):

        LOGGER.debug('receiving data')

        if self.sample_id is None:
            self.sample_id = self.transfer_variable.random_sample_id.get(idx=0)
        if self.host_mask is None:
            self.host_mask = self.transfer_variable.selected_sample_index.get(idx=0)

        local_ref_vec = self._get_reference_vec(data_inst, feat_num=len(x))
        if self.full_explain:
            local_x_mat = self._h_func(self.host_mask, x, local_ref_vec)
            host_table = self._make_sample_table(local_x_mat, x, local_ref_vec)
        else:
            host_table = self._make_non_full_explain_sample_table(x, local_ref_vec)
        LOGGER.debug('start predict')
        self.model.predict(host_table)
        LOGGER.debug('predict done')

    def explain_row(self, x, data_inst):

        if self.role == consts.GUEST:

            if self.full_explain and self.sync_host_feat_num:
                host_feat_num_list = self.transfer_variable.host_feat_num.get(idx=-1)
                self.host_side_feat_num = sum(host_feat_num_list)
                self.host_side_feat_list = host_feat_num_list
                self.sync_host_feat_num = False
                LOGGER.debug('host side anonymous feat num is {}'.format(self.host_side_feat_num))

            host_num = len(self.component_properties.host_party_idlist)
            shap_rs = self.guest_explain_row(x, data_inst, host_num=host_num)
            return shap_rs

        elif self.role == consts.HOST:

            if self.full_explain and self.sync_host_feat_num:
                self.transfer_variable.host_feat_num.remote(len(data_inst.schema['header']))
                self.sync_host_feat_num = False

            self.host_explain_row(x, data_inst)
            return None
        
    def explain(self, data_inst, n=500):

        self.schema = copy.deepcopy(data_inst.schema)
        self.table_partitions = data_inst.partitions
        # test example
        X = data_inst.take(n)
        for inst in X:
            x = inst[1].features
            rs = self.explain_row(x, data_inst)
        
    
        
        
