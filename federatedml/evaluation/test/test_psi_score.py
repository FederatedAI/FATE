from federatedml.evaluation.metrics.classification_metric \
    import PSI, map_ndarray_to_dtable

from federatedml.evaluation.metric_interface import MetricInterface

import numpy as np
import pandas as pd

from arch.api import session

import time

from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.param.feature_binning_param import FeatureBinningParam

session.init("test", 0)

psi = PSI()

interface = MetricInterface(1, 'binary')

expected = np.array([7, 8, 7, 9, 11, 11, 10, 12, 11, 14])
actual = np.array([8, 10, 9, 13, 11, 10, 9, 10, 11, 9])

# total_psi, scores = psi.psi_score(expected, actual)

# nums = np.array([np.random.normal() for i in range(1500)])
# nums2 = np.array([np.random.normal() for i in range(1000)])

# expected_score = pd.DataFrame(nums, columns=['score'])
# actual_score = pd.DataFrame(nums2, columns=['score'])

nums = np.array([np.random.normal() for i in range(1500)])
pos_labels = (nums > 0.5) + 0
nums2 = np.array([np.random.normal() for i in range(1000)])
pos_labels_2 = (nums2 > 0.5) + 0

# index = 0
# expected_score.to_csv('score_1_{}.csv'.format(2), index=False)
# actual_score.to_csv('score_2_{}.csv'.format(2), index=False)

# np.random.shuffle(nums)

# rs = np.quantile(nums, [i*0.05 for i in range(20)] + [1.0], interpolation='nearest')

# quantiles = psi.quantile_binning(nums)
#
# cut_rs = pd.cut(nums, quantiles)

a, b, c, d, e, f, g, h, j = interface.psi(nums, nums2, pos_labels, pos_labels_2, debug=False)

# for i, s, e1, e2, a1, a2 in zip(g, a, c, d, e, f):
#     print(i, s, e1, e2, a1, a2)
#
# psi.compute(train_scores=nums, train_labels=pos_labels, validate_scores=nums2, validate_labels=pos_labels_2)


# np_cut = pd.cut(nums, [0, 10, 20, 30, 30, 30, 30, 100], duplicates='drop')
# np_cut = np_cut.value_counts()

# quantile = [i*0.05 for i in range(20)]
# quantile.append(1.0)

# rs = map_ndarray_to_dtable(nums)

# a = list(rs.collect())
# for i in a:
#     print(i[1].features)

# param = FeatureBinningParam(bin_num=20, error=0.0000001)
# quantile = QuantileBinning(params=param, allow_duplicate=True)
# splits = quantile.fit_split_points(rs)

