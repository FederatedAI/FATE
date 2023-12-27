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

import inspect
from torch import optim
from fate.components.components.nn.hook_code.extract_torch_modules import extract_init_param, Required
from torch.optim.optimizer import required


def code_assembly(param, nn_class):
    para_str = ""
    non_default_param = ""
    init_str = """"""
    special_param = ""
    for k, v in param.items():
        if k == "params":
            k = "params"
            v = None
            special_param = k + "=" + str(v) + ", "
            continue
        else:
            new_para = "\n        self.param_dict['{}'] = {}".format(k, k)
            init_str += new_para

        if isinstance(v, Required) or v == required:
            non_default_param += str(k)
            non_default_param += ", "
            continue

        para_str += str(k)
        if isinstance(v, str):
            para_str += "='{}'".format(v)
        else:
            para_str += "={}".format(str(v))
        para_str += ", "

    para_str = non_default_param + special_param + para_str

    init_ = """
    def __init__(self, {}):
        FateTorchOptimizer.__init__(self){}
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.{}.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except:
            return 'Optimizer {} without initiated parameters'.format(type(self).__name__)

    """.format(
        para_str, init_str, nn_class, nn_class
    )

    code = """
class {}(optim.{}, FateTorchOptimizer):
        {}
    """.format(
        nn_class, nn_class, init_
    )

    return code


if __name__ == "__main__":
    memb = inspect.getmembers(optim)

    module_str = """"""
    for k, v in memb:
        if inspect.isclass(v) and k != "Optimizer":
            param = extract_init_param(v)
            code = code_assembly(param, k)
            module_str += code

    open("../torch/optim.py", "w").write(module_str)
