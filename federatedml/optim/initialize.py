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

import numpy as np


class Initializer:

    def zeros(self, data_shape):
        inits = np.zeros(data_shape)
        return inits

    def random_normal(self, data_shape):
        inits = np.random.randn(data_shape)
        return inits

    def random_uniform(self, data_shape):
        inits = np.random.rand(data_shape)
        return inits

    def constant(self, data_shape, const):

        inits = np.ones(data_shape) * const
        return inits

    def ones(self, data_shape):
        inits = np.ones(data_shape)
        return inits

    def init_model(self, model_shape, init_params):
        init_method = init_params.init_method
        fit_intercept = init_params.fit_intercept

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
            w = self.zeros(model_shape)
        elif init_method == 'const':
            init_const = init_params.init_const
            w = self.constant(model_shape, const=init_const)
        else:
            raise NotImplementedError("Initial method cannot be recognized: {}".format(init_method))
        # if fit_intercept:
        #     coef_ = w[:-1]
        #     intercept_ = w[-1]
        # else:
        #     coef_ = w
        #     intercept_ = 0
        # return coef_, intercept_
        return w
