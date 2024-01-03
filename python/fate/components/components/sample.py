#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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

from typing import Union, Mapping

from fate.arch import Context
from fate.components.core import GUEST, HOST, Role, cpn, params
from fate.ml.model_selection.sample import SampleModuleGuest, SampleModuleHost


@cpn.component(roles=[GUEST, HOST], provider="fate")
def sample(
    ctx: Context,
    role: Role,
    input_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    replace: cpn.parameter(type=bool, default=False, desc="whether allow sampling with replacement, default False"),
    frac: cpn.parameter(
        type=Union[
            params.confloat(gt=0.0), Mapping[Union[params.conint(), params.confloat()], params.confloat(gt=0.0)]
        ],
        default=None,
        optional=True,
        desc="if mode equals to random, it should be a float number greater than 0,"
        "otherwise a dict of pairs like [label_i, sample_rate_i],"
        "e.g. {0: 0.5, 1: 0.8, 2: 0.3}, any label unspecified in dict will not be sampled,"
        "default: 1.0, cannot be used with n",
    ),
    n: cpn.parameter(
        type=params.conint(gt=0),
        default=None,
        optional=True,
        desc="exact sample size, it should be an int greater than 0, " "default: None, cannot be used with frac",
    ),
    random_state: cpn.parameter(type=params.conint(ge=0), default=None, desc="random state"),
    hetero_sync: cpn.parameter(
        type=bool,
        default=True,
        desc="whether guest sync sampled data sids with host, "
        "default True for hetero scenario, "
        "should set to False for local and homo scenario",
    ),
    output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
):
    if frac is not None and n is not None:
        raise ValueError(f"n and frac cannot be used at the same time")
    if frac is not None:
        if isinstance(frac, float):
            if frac > 1 and not replace:
                raise ValueError(f"replace has to be set to True when sampling frac greater than 1.")
        elif isinstance(frac, dict):
            for v in frac.values():
                if v > 1 and not replace:
                    raise ValueError(f"replace has to be set to True when sampling frac greater than 1.")
    if n is None and frac is None:
        frac = 1.0
    # check if local but federated sample
    if hetero_sync and len(ctx.parties.ranks) < 2:
        raise ValueError(f"federated sample can only be called when both 'guest' and 'host' present. Please check")
    sub_ctx = ctx.sub_ctx("train")
    if role.is_guest:
        module = SampleModuleGuest(replace=replace, frac=frac, n=n, random_state=random_state, hetero_sync=hetero_sync)
    elif role.is_host:
        module = SampleModuleHost(replace=replace, frac=frac, n=n, random_state=random_state, hetero_sync=hetero_sync)
    else:
        raise ValueError(f"unknown role")
    input_data = input_data.read()
    original_count = {}
    if input_data.label is not None:
        binarized_label = input_data.label.get_dummies()
        for label_name in binarized_label.schema.columns:
            label_count = binarized_label[label_name].sum().to_list()[0]
            true_label_name = int(label_name.split("_")[1])
            original_count[true_label_name] = label_count
            if isinstance(frac, dict):
                if true_label_name not in frac.keys():
                    frac[true_label_name] = 1.0
        module.frac = frac

    sampled_data = module.fit(sub_ctx, input_data)
    sample_result_summary = {"total": {"original_count": input_data.shape[0], "sampled_count": sampled_data.shape[0]}}
    if input_data.label is not None:
        original_binzied_label = input_data.label.get_dummies()
        sampled_binarized_label = sampled_data.label.get_dummies()
        for label_name in binarized_label.schema.columns:
            original_label_count = original_binzied_label[label_name].sum().to_list()[0]
            sampled_label_count = sampled_binarized_label[label_name].sum().to_list()[0]
            label_summary = {"original_count": original_label_count, "sampled_count": sampled_label_count}
            sample_result_summary[label_name.split("_")[1]] = label_summary

    ctx.metrics.log_metrics(sample_result_summary, "summary")

    output_data.write(sampled_data)
