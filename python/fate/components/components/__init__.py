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
from .evaluation import evaluation
from .feature_scale import feature_scale
from .hetero_feature_binning import hetero_feature_binning
from .hetero_feature_selection import hetero_feature_selection
from .intersection import intersection
from .reader import reader
from .statistics import statistics
from .union import union

BUILDIN_COMPONENTS = [reader, intersection,
                      feature_scale, hetero_feature_binning,
                      evaluation, statistics, hetero_feature_selection,
                      union]
