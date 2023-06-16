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

import json
from typing import List

from fate.components import (
    GUEST,
    HOST,
    DatasetArtifact,
    Input,
    ModelArtifact,
    Output,
    Role,
    cpn,
    params
)


@cpn.component(roles=[GUEST, HOST])
def hetero_feature_selection(ctx, role):
    ...


@hetero_feature_selection.train()
@cpn.artifact("train_data", type=Input[DatasetArtifact], roles=[GUEST, HOST])
@cpn.artifact("input_statistic_model", type=Input[ModelArtifact], roles=[GUEST, HOST], optional=True)
@cpn.artifact("input_binning_model", type=Input[ModelArtifact], roles=[GUEST, HOST], optional=True)
@cpn.parameter("method", type=List[params.string_choice(["manual", "binning", "statistic"])],
               default=["manual"], optional=False,
               desc="selection method, options: {manual, binning, statistic}")
@cpn.parameter("select_col", type=List[str], default=None,
               desc="list of column names to be selected, if None, all columns will be considered")
@cpn.parameter("iv_param", type=params.iv_filter_param(),
               default=params.IVFilterParam(metrics="iv", take_high=True,
                                            threshold=1, filter_type="threshold", host_thresholds=1,
                                            host_take_high=True,
                                            select_federated=True),
               desc="binning filter param")
@cpn.parameter("statistic_param", type=params.statistic_filter_param(),
               default=params.StatisticFilterParam(metrics="mean",
                                                   threshold=1, filter_type="threshold", take_high=True),
               desc="statistic filter param")
@cpn.parameter("manual_param", type=params.manual_filter_param(),
               default=params.ManualFilterParam(filter_out_col=[], keep_col=[]),
               desc="note that manual filter will always be processed as the last filter")
@cpn.parameter("keep_one", type=bool, default=True, desc="whether to keep at least one feature among `select_col`")
@cpn.parameter("use_anonymous", type=bool, default=False,
               desc="bool, whether interpret `select_col` & `filter_out_col` & `keep_col` as anonymous column names")
@cpn.artifact("train_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
@cpn.artifact("output_model", type=Output[ModelArtifact], roles=[GUEST, HOST])
def feature_selection_train(
        ctx,
        role: Role,
        train_data,
        input_statistic_model,
        input_binning_model,
        method,
        select_col,
        iv_param,
        statistic_param,
        manual_param,
        keep_one,
        use_anonymous,
        train_output_data,
        output_model,
):
    train(ctx, role, train_data, train_output_data, input_binning_model, input_statistic_model,
          output_model, method, select_col, iv_param, statistic_param, manual_param,
          keep_one, use_anonymous)


@hetero_feature_selection.predict()
@cpn.artifact("input_model", type=Input[ModelArtifact], roles=[GUEST, HOST])
@cpn.artifact("test_data", type=Input[DatasetArtifact], optional=False, roles=[GUEST, HOST])
@cpn.artifact("test_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
def feature_selection_predict(
        ctx,
        role: Role,
        test_data,
        input_model,
        test_output_data,
):
    predict(ctx, input_model, test_data, test_output_data, role)


def train(ctx, role, train_data, train_output_data, input_binning_model, input_statistic_model,
          output_model, method, select_col, iv_param, statistic_param, manual_param,
          keep_one, use_anonymous):
    from fate.ml.feature_selection import HeteroSelectionModuleHost, HeteroSelectionModuleGuest

    with ctx.sub_ctx("train") as sub_ctx:
        isometric_model_dict = {}
        if input_binning_model:
            with input_binning_model as model_reader:
                model = model_reader.read_model()
            model_type = json.loads(model["model_meta"]).get("model_type")
            if model_type != "binning":
                raise ValueError(f"model type: {model_type} is not binning, but {model_type}")
            isometric_model_dict["binning"] = model
        if input_statistic_model:
            with input_statistic_model as model_reader:
                model = model_reader.read_model()
            # temp code block
            model_type = json.loads(model["model_meta"]).get("model_type")
            if model_type != "statistic":
                raise ValueError(f"model type: {model_type} is not statistic, but {model_type}")
            # temp code block end
            isometric_model_dict["statistic"] = model

        # logger.info(f"input model: {isometric_model_dict}")

        train_data = sub_ctx.reader(train_data).read_dataframe().data
        columns = train_data.schema.columns.to_list()
        if use_anonymous:
            anonymous_columns = train_data.schema.anonymous_columns.to_list()
            if select_col is not None:
                select_col = [columns[anonymous_columns.index(col)] for col in select_col]
            if manual_param.filter_out_col is not None:
                filter_out_col = [columns[anonymous_columns.index(col)] for col in manual_param.filter_out_col]
                manual_param.filter_out_col = filter_out_col
            if manual_param.keep_col is not None:
                keep_col = [columns[anonymous_columns.index(col)] for col in manual_param.keep_col]
                manual_param.keep_col = keep_col

        if role.is_guest:
            selection = HeteroSelectionModuleGuest(method, select_col, isometric_model_dict,
                                                   iv_param, statistic_param, manual_param,
                                                   keep_one)
        elif role.is_host:
            selection = HeteroSelectionModuleHost(method, select_col, isometric_model_dict,
                                                  iv_param, statistic_param, manual_param,
                                                  keep_one)
        selection.fit(sub_ctx, train_data)
        model = selection.to_model()
        with output_model as model_writer:
            model_writer.write_model("feature_selection", model, metadata={"method": method})

    with ctx.sub_ctx("predict") as sub_ctx:
        output_data = train_data
        if method is not None:
            output_data = selection.transform(sub_ctx, train_data)
        sub_ctx.writer(train_output_data).write_dataframe(output_data)


def predict(ctx, input_model, test_data, test_output_data, role):
    from fate.ml.feature_selection import HeteroSelectionModuleHost, HeteroSelectionModuleGuest

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()
        if role.is_guest:
            selection = HeteroSelectionModuleGuest.from_model(model)
        elif role.is_host:
            selection = HeteroSelectionModuleHost.from_model(model)

        model_meta = model["meta_data"]
        method = model_meta["method"]
        selection.method = method
        test_data = sub_ctx.reader(test_data).read_dataframe().data

        output_data = test_data
        if method is not None:
            output_data = selection.transform(sub_ctx, test_data)
        """
        # temp code start
        test_data = sub_ctx.reader(test_data).read_dataframe().data
        output_data = selection.transform(sub_ctx, test_data)
        # temp code end
        """
        sub_ctx.writer(test_output_data).write_dataframe(output_data)
