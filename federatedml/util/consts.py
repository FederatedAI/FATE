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
PAILLIER = 'Paillier'
RANDOM_PADS = "RandomPads"
NONE = "None"
AFFINE = 'Affine'
ITERATIVEAFFINE = 'IterativeAffine'
L1_PENALTY = 'L1'
L2_PENALTY = 'L2'

FLOAT_ZERO = 1e-6

PARAM_MAXDEPTH = 5
MAX_CLASSNUM = 1000
MIN_BATCH_SIZE = 10
SPARSE_VECTOR = "SparseVector"

HETERO = "hetero"
HOMO = "homo"

RAW = "raw"
RSA = "rsa"

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
MAX_FEDERATED_NODES = 2 ** 10

TRAIN_EVALUATE = 'train_evaluate'
VALIDATE_EVALUATE = 'validate_evaluate'

# Feature engineering
G_BIN_NUM = 10
DEFAULT_COMPRESS_THRESHOLD = 10000
DEFAULT_HEAD_SIZE = 10000
DEFAULT_RELATIVE_ERROR = 0.001
ONE_HOT_LIMIT = 1024   # No more than 10 possible values

QUANTILE = 'quantile'
BUCKET = 'bucket'
OPTIMAL = 'optimal'

# Feature selection methods
UNIQUE_VALUE = 'unique_value'
IV_VALUE_THRES = 'iv_value_thres'
IV_PERCENTILE = 'iv_percentile'
COEFFICIENT_OF_VARIATION_VALUE_THRES = 'coefficient_of_variation_value_thres'
# COEFFICIENT_OF_VARIATION_PERCENTILE = 'coefficient_of_variation_percentile'
OUTLIER_COLS = 'outlier_cols'
MANUALLY_FILTER = 'manually'

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
SHA256 = 'sha256'
INTERSECT_CACHE_TAG = 'Za'