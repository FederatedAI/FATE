#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import copy

import numpy as np
from federatedml.feature.binning.base_binning import BaseBinning
from federatedml.feature.binning.bin_inner_param import BinInnerParam
from federatedml.feature.binning.bin_result import MultiClassBinResult
from federatedml.feature.binning.bucket_binning import BucketBinning
from federatedml.feature.binning.iv_calculator import IvCalculator
from federatedml.feature.binning.optimal_binning.optimal_binning import \
    OptimalBinning
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.feature.fate_element_type import NoneType
from federatedml.feature.sparse_vector import SparseVector
from federatedml.model_base import ModelBase
from federatedml.param.feature_binning_param import \
    HeteroFeatureBinningParam as FeatureBinningParam
from federatedml.protobuf.generated import (feature_binning_meta_pb2,
                                            feature_binning_param_pb2)
from federatedml.statistic.data_overview import (get_anonymous_header,
                                                 get_header)
from federatedml.transfer_variable.transfer_class.hetero_feature_binning_transfer_variable import \
    HeteroFeatureBinningTransferVariable
from federatedml.util import LOGGER, abnormal_detection, consts
from federatedml.util.anonymous_generator_util import Anonymous
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.util.schema_check import assert_schema_consistent

MODEL_PARAM_NAME = 'FeatureBinningParam'
MODEL_META_NAME = 'FeatureBinningMeta'


class BaseFeatureBinning(ModelBase):
    """
    Do binning method through guest and host

    """

    def __init__(self):
        super(BaseFeatureBinning, self).__init__()
        self.transfer_variable = HeteroFeatureBinningTransferVariable()  # 异构特征guest和host交流变量
        self.binning_obj: BaseBinning = None
        self.header = None
        self.anonymous_header = None
        self.training_anonymous_header = None
        self.schema = None  # 数据库
        self.host_results = []
        self.transform_host_results = []
        self.transform_type = None

        self.model_param = FeatureBinningParam()  # 模型参数,主要为分箱算法用到的具体参数
        self.bin_inner_param = BinInnerParam()  # 分箱过程中的参数
        self.bin_result = MultiClassBinResult(labels=[0, 1])  # 二分类的分箱结果
        self.transform_bin_result = MultiClassBinResult(labels=[0, 1])  # 转换后的二分类分箱结果
        self.has_missing_value = False
        self.labels = []

        self._stage = "fit"

    # 初始化模型
    def _init_model(self, params: FeatureBinningParam):
        self.model_param = params

        self.transform_type = self.model_param.transform_param.transform_type

        if self.role == consts.HOST:
            if self.transform_type == "woe":
                raise ValueError("Host party do not support woe transform now.")

        if self.model_param.method == consts.QUANTILE:
            self.binning_obj = QuantileBinning(self.model_param)  # 量化分箱(无监督分箱)
        elif self.model_param.method == consts.BUCKET:
            self.binning_obj = BucketBinning(self.model_param)  # 桶分箱(有监督分箱)
        elif self.model_param.method == consts.OPTIMAL:  # 优化分箱(无监督分箱)
            if self.role == consts.HOST:  # host为量化分箱
                self.model_param.bin_num = self.model_param.optimal_binning_param.init_bin_nums  # 初始化分箱数目
                self.binning_obj = QuantileBinning(self.model_param)  # 再量化分箱(host没有label，只能无监督分箱)
            else:  # guest为优化分箱
                self.binning_obj = OptimalBinning(self.model_param)  # 优化分箱
        else:
            raise ValueError("Binning method: {} is not supported yet".format(self.model_param.method))

        self.iv_calculator = IvCalculator(self.model_param.adjustment_factor,
                                          role=self.role,
                                          party_id=self.component_properties.local_partyid)  # 初始化iv计算函数
        # self.binning_obj.set_role_party(self.role, self.component_properties.local_partyid)

    @staticmethod
    def data_format_transform(row):
        """
        transform data into sparse format{label feature_id:value}
        """

        if type(row.features).__name__ != consts.SPARSE_VECTOR:
            feature_shape = row.features.shape[0]
            indices = []
            data = []

            for i in range(feature_shape):
                if np.isnan(row.features[i]):
                    indices.append(i)
                    data.append(NoneType())
                elif np.abs(row.features[i]) < consts.FLOAT_ZERO:
                    continue
                else:
                    indices.append(i)
                    data.append(row.features[i])

            new_row = copy.deepcopy(row)
            new_row.features = SparseVector(indices, data, feature_shape)
            return new_row
        else:
            sparse_vec = row.features.get_sparse_vector()
            replace_key = []
            for key in sparse_vec:
                if sparse_vec.get(key) == NoneType() or np.isnan(sparse_vec.get(key)):
                    replace_key.append(key)

            if len(replace_key) == 0:
                return row
            else:
                new_row = copy.deepcopy(row)
                new_sparse_vec = new_row.features.get_sparse_vector()
                for key in replace_key:
                    new_sparse_vec[key] = NoneType()
                return new_row

    # 设置分箱过程中的参数
    def _setup_bin_inner_param(self, data_instances, params):
        if self.schema is not None:
            return

        self.header = get_header(data_instances)
        self.anonymous_header = get_anonymous_header(data_instances)
        LOGGER.debug("_setup_bin_inner_param, get header length: {}".format(len(self.header)))

        self.schema = data_instances.schema
        self.bin_inner_param.set_header(self.header, self.anonymous_header)
        if params.bin_indexes == -1:
            self.bin_inner_param.set_bin_all()
        else:
            self.bin_inner_param.add_bin_indexes(params.bin_indexes)
            self.bin_inner_param.add_bin_names(params.bin_names)

        self.bin_inner_param.add_category_indexes(params.category_indexes)
        self.bin_inner_param.add_category_names(params.category_names)

        if params.transform_param.transform_cols == -1:
            self.bin_inner_param.set_transform_all()
        else:
            self.bin_inner_param.add_transform_bin_indexes(params.transform_param.transform_cols)
            self.bin_inner_param.add_transform_bin_names(params.transform_param.transform_names)
        self.binning_obj.set_bin_inner_param(self.bin_inner_param)

    # 将guest分箱中特征值转换为woe编码
    @assert_io_num_rows_equal
    @assert_schema_consistent
    def transform_data(self, data_instances):
        self._setup_bin_inner_param(data_instances, self.model_param)
        if self.transform_type != "woe":
            data_instances = self.binning_obj.transform(data_instances, self.transform_type)
        elif self.role == consts.HOST:
            raise ValueError("Woe transform is not available for host parties.")
        else:
            data_instances = self.iv_calculator.woe_transformer(data_instances, self.bin_inner_param,
                                                                self.bin_result)
        self.set_schema(data_instances)
        self.data_output = data_instances
        return data_instances

    # 从protobuf中获取元数据
    def _get_meta(self):
        """
        method
        compress_thres
        head_size
        error
        bin_num
        cols
        adjustment_factor
        local_only
        need_run
        transform_param
        skip_static
        """
        # col_list = [str(x) for x in self.cols]

        transform_param = feature_binning_meta_pb2.TransformMeta(
            transform_cols=self.bin_inner_param.transform_bin_indexes,
            transform_type=self.model_param.transform_param.transform_type
        )

        meta_protobuf_obj = feature_binning_meta_pb2.FeatureBinningMeta(
            method=self.model_param.method,
            compress_thres=self.model_param.compress_thres,
            head_size=self.model_param.head_size,
            error=self.model_param.error,
            bin_num=self.model_param.bin_num,
            cols=self.bin_inner_param.bin_names,
            adjustment_factor=self.model_param.adjustment_factor,
            local_only=self.model_param.local_only,
            need_run=self.need_run,
            transform_param=transform_param,
            skip_static=self.model_param.skip_static
        )
        return meta_protobuf_obj

    # 从protobuf获取参数
    def _get_param(self):
        """

        """
        split_points_result = self.binning_obj.bin_results.split_results

        multi_class_result = self.bin_result.generated_pb_list(split_points_result)
        # LOGGER.debug(f"split_points_result: {split_points_result}")
        host_multi_class_result = []
        host_single_results = []

        anonymous_dict_list = []
        if self._stage == "transform" and self._check_lower_version_anonymous():
            if self.role == consts.GUEST:
                anonymous_dict_list = self.transfer_variable.host_anonymous_header_dict.get(idx=-1)
            elif self.role == consts.HOST:
                anonymous_dict = dict(zip(self.training_anonymous_header, self.anonymous_header))
                self.transfer_variable.host_anonymous_header_dict.remote(
                    anonymous_dict,
                    role=consts.GUEST,
                    idx=0
                )

        for idx, host_res in enumerate(self.host_results):
            if not anonymous_dict_list:
                host_multi_class_result.extend(host_res.generated_pb_list())
                host_single_results.append(host_res.bin_results[0].generated_pb())
            else:
                updated_anonymous_header = anonymous_dict_list[idx]
                host_res.update_anonymous(updated_anonymous_header)
                host_multi_class_result.extend(host_res.generated_pb_list())
                host_single_results.append(host_res.bin_results[0].generated_pb())

        has_host_result = True if len(host_multi_class_result) else False
        multi_pb = feature_binning_param_pb2.MultiClassResult(
            results=multi_class_result,
            labels=[str(x) for x in self.labels],
            host_results=host_multi_class_result,
            host_party_ids=[str(x) for x in self.component_properties.host_party_idlist],
            has_host_result=has_host_result
        )
        if self._stage == "fit":
            result_obj = feature_binning_param_pb2. \
                FeatureBinningParam(binning_result=multi_class_result[0],
                                    host_results=host_single_results,
                                    header=self.header,
                                    header_anonymous=self.anonymous_header,
                                    model_name=consts.BINNING_MODEL,
                                    multi_class_result=multi_pb)
        else:
            transform_multi_class_result = self.transform_bin_result.generated_pb_list(split_points_result)
            transform_host_single_results = []
            transform_host_multi_class_result = []
            for host_res in self.transform_host_results:
                transform_host_multi_class_result.extend(host_res.generated_pb_list())
                transform_host_single_results.append(host_res.bin_results[0].generated_pb())

            transform_multi_pb = feature_binning_param_pb2.MultiClassResult(
                results=transform_multi_class_result,
                labels=[str(x) for x in self.labels],
                host_results=transform_host_multi_class_result,
                host_party_ids=[str(x) for x in self.component_properties.host_party_idlist],
                has_host_result=has_host_result
            )

            result_obj = feature_binning_param_pb2. \
                FeatureBinningParam(binning_result=multi_class_result[0],
                                    host_results=host_single_results,
                                    header=self.header,
                                    header_anonymous=self.anonymous_header,
                                    model_name=consts.BINNING_MODEL,
                                    multi_class_result=multi_pb,
                                    transform_binning_result=transform_multi_class_result[0],
                                    transform_host_results=transform_host_single_results,
                                    transform_multi_class_result=transform_multi_pb)

        return result_obj

    # 加载模型
    def load_model(self, model_dict):

        # 将元数据和模型参数读入当前类
        model_param = list(model_dict.get('model').values())[0].get(MODEL_PARAM_NAME)
        model_meta = list(model_dict.get('model').values())[0].get(MODEL_META_NAME)

        self.bin_inner_param = BinInnerParam()
        multi_class_result = model_param.multi_class_result
        self.labels = list(map(int, multi_class_result.labels))
        # if not self.labels:
        #     self.labels = [0, 1]
        if self.labels:
            self.bin_result = MultiClassBinResult.reconstruct(list(multi_class_result.results), self.labels)

        # instance类型判断
        assert isinstance(model_meta, feature_binning_meta_pb2.FeatureBinningMeta)
        assert isinstance(model_param, feature_binning_param_pb2.FeatureBinningParam)

        # 表头设置
        self.header = list(model_param.header)
        self.training_anonymous_header = list(model_param.header_anonymous)
        self.bin_inner_param.set_header(self.header, self.training_anonymous_header)

        self.bin_inner_param.add_transform_bin_indexes(list(model_meta.transform_param.transform_cols))
        self.bin_inner_param.add_bin_names(list(model_meta.cols))
        self.transform_type = model_meta.transform_param.transform_type

        # 调用分箱方法进行分箱
        bin_method = str(model_meta.method)
        if bin_method == consts.QUANTILE:
            self.binning_obj = QuantileBinning(params=model_meta)  # 量化分箱
        elif bin_method == consts.OPTIMAL:
            self.binning_obj = OptimalBinning(params=model_meta)  # 优化分箱
        else:
            self.binning_obj = BucketBinning(params=model_meta)  # 桶分箱

        # self.binning_obj.set_role_party(self.role, self.component_properties.local_partyid)
        self.binning_obj.set_bin_inner_param(self.bin_inner_param)

        # 保存分箱结果
        split_results = dict(model_param.binning_result.binning_result)
        for col_name, sr_pb in split_results.items():
            split_points = list(sr_pb.split_points)
            self.binning_obj.bin_results.put_col_split_points(col_name, split_points)

        # self.binning_obj.bin_results.reconstruct(model_param.binning_result)

        # host的分箱结果
        self.host_results = []
        host_pbs = list(model_param.multi_class_result.host_results)
        if len(host_pbs):
            if len(self.labels) == 2:
                for host_pb in host_pbs:
                    self.host_results.append(MultiClassBinResult.reconstruct(
                        host_pb, self.labels))
            else:
                assert len(host_pbs) % len(self.labels) == 0
                i = 0
                while i < len(host_pbs):
                    this_pbs = host_pbs[i: i + len(self.labels)]
                    self.host_results.append(MultiClassBinResult.reconstruct(this_pbs, self.labels))
                    i += len(self.labels)

        """
        if list(model_param.header_anonymous):
            self.anonymous_header = list(model_param.anonymous_header)
        """

        self._stage = "transform"

    # 导出模型（包括：模型元数据和模型参数）
    def export_model(self):
        if self.model_output is not None:
            return self.model_output

        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            MODEL_META_NAME: meta_obj,
            MODEL_PARAM_NAME: param_obj
        }
        self.model_output = result
        return result

    # 保存数据
    def save_data(self):
        return self.data_output

    # 设置数据schema
    def set_schema(self, data_instance):
        self.schema['header'] = self.header
        data_instance.schema = self.schema
        # LOGGER.debug("After Binning, when setting schema, schema is : {}".format(data_instance.schema))

    # 异常检测
    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)
        self.check_schema_content(data_instances.schema)

    # 老版本数据匿名化检查
    def _check_lower_version_anonymous(self):
        return not self.training_anonymous_header or \
            Anonymous.is_old_version_anonymous_header(self.training_anonymous_header)
