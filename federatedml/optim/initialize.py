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

from collections.abc import Iterable

import numpy as np

from arch.api.utils import log_utils
from federatedml.statistic import statics

LOGGER = log_utils.getLogger()


class Initializer(object):
    def zeros(self, data_shape, fit_intercept, data_instances):
        """
        If fit intercept, use the following formula to initialize b can get a faster converge rate
            b = log(P(1)/P(0))
        """

        inits = np.zeros(data_shape)
        if fit_intercept and data_instances is not None:
            static_obj = statics.MultivariateStatisticalSummary(data_instances, cols_index=-1)
            label_historgram = static_obj.get_label_histogram()
            LOGGER.debug("label_histogram is : {}".format(label_historgram))
            one_count = label_historgram.get(1)
            zero_count = label_historgram.get(0, 0) + label_historgram.get(-1, 0)
            init_intercept = np.log((one_count / zero_count))
            inits[-1] = init_intercept
        return inits

    def random_normal(self, data_shape):
        if isinstance(data_shape, Iterable):
            inits = np.random.randn(*data_shape)
        else:
            inits = np.random.randn(data_shape)
        return inits

    def random_uniform(self, data_shape):
        if isinstance(data_shape, Iterable):
            inits = np.random.rand(*data_shape)
        else:
            inits = np.random.rand(data_shape)
        return inits

    def constant(self, data_shape, const):
        inits = np.ones(data_shape) * const
        return inits

    def ones(self, data_shape):
        inits = np.ones(data_shape)
        return inits

    def init_model(self, model_shape, init_params, data_instance=None):
        init_method = init_params.init_method
        fit_intercept = init_params.fit_intercept

        random_seed = init_params.random_seed
        np.random.seed(random_seed)

        if fit_intercept:
            if isinstance(model_shape, int):
                model_shape += 1
            else:
                new_shape = []
                for ds in model_shape:
                    new_shape.append(ds + 1)
                model_shape = tuple(new_shape)

        if init_method == 'random_normal':
            w = self.random_normal(model_shape)
        elif init_method == 'random_uniform':
            w = self.random_uniform(model_shape)
        elif init_method == 'ones':
            w = self.ones(model_shape)
        elif init_method == 'zeros':
            w = self.zeros(model_shape, fit_intercept, data_instance)
        elif init_method == 'const':
            init_const = init_params.init_const
            w = self.constant(model_shape, const=init_const)
        else:
            raise NotImplementedError("Initial method cannot be recognized: {}".format(init_method))
        LOGGER.debug("Inited model is :{}".format(w))
        return w
