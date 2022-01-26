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

ARBITER = 'arbiter'
HOST = 'host'
GUEST = 'guest'


MODEL_AGG = "model_agg"
GRAD_AGG = "grad_agg"

BINARY = 'binary'
MULTY = 'multi'
CLASSIFICATION = "classification"
REGRESSION = 'regression'
CLUSTERING = 'clustering'
ONE_VS_REST = 'one_vs_rest'
PAILLIER = 'Paillier'
RANDOM_PADS = "RandomPads"
NONE = "None"
AFFINE = 'Affine'
ITERATIVEAFFINE = 'IterativeAffine'
RANDOM_ITERATIVEAFFINE = 'RandomIterativeAffine'
L1_PENALTY = 'L1'
L2_PENALTY = 'L2'

FLOAT_ZERO = 1e-8
OVERFLOW_THRESHOLD = 1e8
OT_HAUCK = 'OT_Hauck'
CE_PH = 'CommutativeEncryptionPohligHellman'
XOR = 'xor'
AES = 'aes'

PARAM_MAXDEPTH = 5
MAX_CLASSNUM = 1000
MIN_BATCH_SIZE = 10
SPARSE_VECTOR = "SparseVector"

HETERO = "hetero"
HOMO = "homo"

RAW = "raw"
RSA = "rsa"
DH = "dh"

# evaluation
AUC = "auc"
KS = "ks"
LIFT = "lift"
GAIN = "gain"
PRECISION = "precision"
RECALL = "recall"
ACCURACY = "accuracy"
EXPLAINED_VARIANCE = "explained_variance"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
MEAN_SQUARED_ERROR = "mean_squared_error"
MEAN_SQUARED_LOG_ERROR = "mean_squared_log_error"
MEDIAN_ABSOLUTE_ERROR = "median_absolute_error"
R2_SCORE = "r2_score"
ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
ROC = "roc"
F1_SCORE = 'f1_score'
CONFUSION_MAT = 'confusion_mat'
PSI = 'psi'
VIF = 'vif'
PEARSON = 'pearson'
FEATURE_IMPORTANCE = 'feature_importance'
QUANTILE_PR = 'quantile_pr'
JACCARD_SIMILARITY_SCORE = 'jaccard_similarity_score'
FOWLKES_MALLOWS_SCORE = 'fowlkes_mallows_score'
ADJUSTED_RAND_SCORE = 'adjusted_rand_score'
DAVIES_BOULDIN_INDEX = 'davies_bouldin_index'
DISTANCE_MEASURE = 'distance_measure'
CONTINGENCY_MATRIX = 'contingency_matrix'

# evaluation alias metric
ALL_METRIC_NAME = [AUC, KS, LIFT, GAIN, PRECISION, RECALL, ACCURACY, EXPLAINED_VARIANCE, MEAN_ABSOLUTE_ERROR,
                   MEAN_SQUARED_ERROR, MEAN_SQUARED_LOG_ERROR, MEDIAN_ABSOLUTE_ERROR, R2_SCORE, ROOT_MEAN_SQUARED_ERROR,
                   ROC, F1_SCORE, CONFUSION_MAT, PSI, QUANTILE_PR, JACCARD_SIMILARITY_SCORE, FOWLKES_MALLOWS_SCORE,
                   ADJUSTED_RAND_SCORE, DAVIES_BOULDIN_INDEX, DISTANCE_MEASURE, CONTINGENCY_MATRIX]

ALIAS = {
    ('l1', 'mae', 'regression_l1'): MEAN_ABSOLUTE_ERROR,
    ('l2', 'mse', 'regression_l2', 'regression'): MEAN_SQUARED_ERROR,
    ('l2_root', 'rmse'): ROOT_MEAN_SQUARED_ERROR,
    ('msle', ): MEAN_SQUARED_LOG_ERROR,
    ('r2', ): R2_SCORE,
    ('acc', ): ACCURACY,
    ('DBI', ): DAVIES_BOULDIN_INDEX,
    ('FMI', ): FOWLKES_MALLOWS_SCORE,
    ('RI', ): ADJUSTED_RAND_SCORE,
    ('jaccard', ): JACCARD_SIMILARITY_SCORE
}

# default evaluation metrics
DEFAULT_BINARY_METRIC = [AUC, KS]
DEFAULT_REGRESSION_METRIC = [ROOT_MEAN_SQUARED_ERROR, MEAN_ABSOLUTE_ERROR]
DEFAULT_MULTI_METRIC = [ACCURACY, PRECISION, RECALL]
DEFAULT_CLUSTER_METRIC = [DAVIES_BOULDIN_INDEX]

# allowed metrics for different tasks
ALL_BINARY_METRICS = [
    AUC,
    KS,
    LIFT,
    GAIN,
    ACCURACY,
    PRECISION,
    RECALL,
    ROC,
    CONFUSION_MAT,
    PSI,
    F1_SCORE,
    QUANTILE_PR
]

ALL_REGRESSION_METRICS = [
    EXPLAINED_VARIANCE,
    MEAN_ABSOLUTE_ERROR,
    MEAN_SQUARED_ERROR,
    MEDIAN_ABSOLUTE_ERROR,
    R2_SCORE,
    ROOT_MEAN_SQUARED_ERROR
]

ALL_MULTI_METRICS = [
    ACCURACY,
    PRECISION,
    RECALL
]
ALL_CLUSTER_METRICS = [
    JACCARD_SIMILARITY_SCORE,
    FOWLKES_MALLOWS_SCORE,
    ADJUSTED_RAND_SCORE,
    DAVIES_BOULDIN_INDEX,
    DISTANCE_MEASURE,
    CONTINGENCY_MATRIX
]

# single value metrics
REGRESSION_SINGLE_VALUE_METRICS = [
    EXPLAINED_VARIANCE,
    MEAN_ABSOLUTE_ERROR,
    MEAN_SQUARED_ERROR,
    MEAN_SQUARED_LOG_ERROR,
    MEDIAN_ABSOLUTE_ERROR,
    R2_SCORE,
    ROOT_MEAN_SQUARED_ERROR,
]

BINARY_SINGLE_VALUE_METRIC = [
    AUC,
    KS
]

MULTI_SINGLE_VALUE_METRIC = [
    PRECISION,
    RECALL,
    ACCURACY
]

CLUSTER_SINGLE_VALUE_METRIC = [
    JACCARD_SIMILARITY_SCORE,
    FOWLKES_MALLOWS_SCORE,
    ADJUSTED_RAND_SCORE,
    DAVIES_BOULDIN_INDEX
]
# workflow
TRAIN_DATA = "train_data"
TEST_DATA = "test_data"

# initialize method
RANDOM_NORMAL = "random_normal"
RANDOM_UNIFORM = 'random_uniform'
ONES = 'ones'
ZEROS = 'zeros'
CONST = 'const'

# decision tree
MAX_SPLIT_NODES = 2 ** 16
MAX_SPLITINFO_TO_COMPUTE = 2 ** 10
NORMAL_TREE = 'normal'
COMPLETE_SECURE_TREE = 'complete_secure'
STD_TREE = 'std'
MIX_TREE = 'mix'
LAYERED_TREE = 'layered'
SINGLE_OUTPUT = 'single_output'
MULTI_OUTPUT = 'multi_output'

TRAIN_EVALUATE = 'train_evaluate'
VALIDATE_EVALUATE = 'validate_evaluate'

# Feature engineering
G_BIN_NUM = 10
DEFAULT_COMPRESS_THRESHOLD = 10000
DEFAULT_HEAD_SIZE = 10000
DEFAULT_RELATIVE_ERROR = 1e-4
ONE_HOT_LIMIT = 1024   # No more than 10 possible values
PERCENTAGE_VALUE_LIMIT = 0.1
SECURE_AGG_AMPLIFY_FACTOR = 1000

QUANTILE = 'quantile'
BUCKET = 'bucket'
OPTIMAL = 'optimal'
VIRTUAL_SUMMARY = 'virtual_summary'
RECURSIVE_QUERY = 'recursive_query'

# Feature selection methods
UNIQUE_VALUE = 'unique_value'
IV_VALUE_THRES = 'iv_value_thres'
IV_PERCENTILE = 'iv_percentile'
IV_TOP_K = 'iv_top_k'
COEFFICIENT_OF_VARIATION_VALUE_THRES = 'coefficient_of_variation_value_thres'
# COEFFICIENT_OF_VARIATION_PERCENTILE = 'coefficient_of_variation_percentile'
OUTLIER_COLS = 'outlier_cols'
MANUALLY_FILTER = 'manually'
PERCENTAGE_VALUE = 'percentage_value'
IV_FILTER = 'iv_filter'
STATISTIC_FILTER = 'statistic_filter'
PSI_FILTER = 'psi_filter'
VIF_FILTER = 'vif_filter'
CORRELATION_FILTER = 'correlation_filter'
SECUREBOOST = 'sbt'
HETERO_SBT_FILTER = 'hetero_sbt_filter'
HOMO_SBT_FILTER = 'homo_sbt_filter'
HETERO_FAST_SBT_FILTER = 'hetero_fast_sbt_filter'
IV = 'iv'

# Selection Pre-model
STATISTIC_MODEL = 'statistic_model'
BINNING_MODEL = 'binning_model'

# imputer
MIN = 'min'
MAX = 'max'
MEAN = 'mean'
DESIGNATED = 'designated'
STR = 'str'
FLOAT = 'float'
INT = 'int'
ORIGIN = 'origin'
MEDIAN = 'median'

# min_max_scaler
NORMAL = 'normal'
CAP = 'cap'
MINMAXSCALE = 'min_max_scale'
STANDARDSCALE = 'standard_scale'
ALL = 'all'
COL = 'col'

# intersection cache
PHONE = 'phone'
IMEI = 'imei'
MD5 = 'md5'
SHA1 = 'sha1'
SHA224 = 'sha224'
SHA256 = 'sha256'
SHA384 = 'sha384'
SHA512 = 'sha512'
SM3 = 'sm3'
INTERSECT_CACHE_TAG = 'Za'

SHARE_INFO_COL_NAME = "share_info"

# statistics
COUNT = 'count'
STANDARD_DEVIATION = 'stddev'
SUMMARY = 'summary'
DESCRIBE = 'describe'
SUM = 'sum'
COVARIANCE = 'cov'
CORRELATION = 'corr'
VARIANCE = 'variance'
COEFFICIENT_OF_VARIATION = 'coefficient_of_variance'
MISSING_COUNT = "missing_count"
MISSING_RATIO = "missing_ratio"
SKEWNESS = 'skewness'
KURTOSIS = 'kurtosis'


# adapters model name
HOMO_SBT = 'homo_sbt'
HETERO_SBT = 'hetero_sbt'
HETERO_FAST_SBT = 'hetero_fast_sbt'
HETERO_FAST_SBT_MIX = 'hetero_fast_sbt_mix'
HETERO_FAST_SBT_LAYERED = 'hetero_fast_sbt_layered'

# tree protobuf model name
HETERO_SBT_GUEST_MODEL = 'HeteroSecureBoostingTreeGuest'
HETERO_SBT_HOST_MODEL = 'HeteroSecureBoostingTreeHost'
HETERO_FAST_SBT_GUEST_MODEL = "HeteroFastSecureBoostingTreeGuest"
HETERO_FAST_SBT_HOST_MODEL = "HeteroFastSecureBoostingTreeHost"
HOMO_SBT_GUEST_MODEL = "HomoSecureBoostingTreeGuest"
HOMO_SBT_HOST_MODEL = "HomoSecureBoostingTreeHost"

# tree decimal round to prevent float error
TREE_DECIMAL_ROUND = 10

# homm sbt backend
MEMORY_BACKEND = 'memory'
DISTRIBUTED_BACKEND = 'distributed'

# column_expand
MANUAL = 'manual'

# scorecard
CREDIT = 'credit'

# sample weight
BALANCED = 'balanced'

# min r base fraction
MIN_BASE_FRACTION = 0.01
MAX_BASE_FRACTION = 0.99

MAX_SAMPLE_OUTPUT_LIMIT = 10 ** 6

# Hetero NN Selective BP Strategy
SELECTIVE_SIZE = 1024

# intersect join methods
INNER_JOIN = "inner_join"
LEFT_JOIN = "left_join"

DEFAULT_KEY_LENGTH = 1024

MIN_HASH_FUNC_COUNT = 4
MAX_HASH_FUNC_COUNT = 32
