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

import json

from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import HeteroFeatureBinning
from pipeline.component import Intersection
from pipeline.component import OneHotEncoder
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.interface import Model


def prettify(response, verbose=True):
    if verbose:
        print(json.dumps(response, indent=4, ensure_ascii=False))
        print()
    return response


def make_add_one_hot_dsl(config, namespace, bin_param, is_multi_host=False):
    parties = config.parties
    guest = parties.guest[0]
    hosts = parties.host

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    guest_eval_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_eval_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    if is_multi_host:
        pipeline.set_roles(guest=guest, host=hosts)
    else:
        pipeline.set_roles(guest=guest, host=hosts[0])

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=hosts[0]).component_param(table=host_train_data)
    if is_multi_host:
        reader_0.get_party_instance(role='host', party_id=hosts[1]).component_param(table=host_train_data)

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=guest_eval_data)
    reader_1.get_party_instance(role='host', party_id=hosts[0]).component_param(table=host_eval_data)
    if is_multi_host:
        reader_1.get_party_instance(role='host', party_id=hosts[1]).component_param(table=host_eval_data)

    # define DataTransform components
    data_transform_0 = DataTransform(name="data_transform_0")  # start component numbering at 0
    data_transform_1 = DataTransform(name="data_transform_1")

    # get DataTransform party instance of guest
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(role='guest', party_id=guest)
    # configure DataTransform for guest
    data_transform_0_guest_party_instance.component_param(with_label=True, output_format="dense")
    # get and configure DataTransform party instance of host
    data_transform_0.get_party_instance(role='host', party_id=hosts[0]).component_param(with_label=False)
    if is_multi_host:
        data_transform_0.get_party_instance(role='host', party_id=hosts[1]).component_param(with_label=False)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")
    intersection_1 = Intersection(name="intersection_1")

    hetero_feature_binning_0 = HeteroFeatureBinning(**bin_param)
    hetero_feature_binning_1 = HeteroFeatureBinning(name='hetero_feature_binning_1')

    one_hot_encoder_0 = OneHotEncoder(name='one_hot_encoder_0',
                                      transform_col_indexes=-1,
                                      transform_col_names=None,
                                      need_run=True)
    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    # set data_transform_1 to replicate model from data_transform_0
    pipeline.add_component(
        data_transform_1, data=Data(
            data=reader_1.output.data), model=Model(
            data_transform_0.output.model))
    # set data input sources of intersection components
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(intersection_1, data=Data(data=data_transform_1.output.data))
    # set train & validate data of hetero_lr_0 component
    pipeline.add_component(hetero_feature_binning_0, data=Data(data=intersection_0.output.data))
    pipeline.add_component(hetero_feature_binning_1, data=Data(data=intersection_1.output.data),
                           model=Model(hetero_feature_binning_0.output.model))

    pipeline.add_component(one_hot_encoder_0, data=Data(data=hetero_feature_binning_0.output.data))

    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # pipeline.fit(work_mode=work_mode)
    return pipeline


def make_normal_dsl(config, namespace, bin_param, dataset='breast', is_multi_host=False,
                    host_dense_output=True):
    parties = config.parties
    guest = parties.guest[0]
    hosts = parties.host

    if dataset == 'breast':
        guest_table_name = 'breast_hetero_guest'
        host_table_name = 'breast_hetero_host'
    elif dataset == 'default_credit':
        guest_table_name = 'default_credit_hetero_guest'
        host_table_name = 'default_credit_hetero_host'
    else:
        raise ValueError(f"dataset: {dataset} cannot be recognized")

    guest_train_data = {"name": guest_table_name, "namespace": f"experiment{namespace}"}
    host_train_data = {"name": host_table_name, "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    if is_multi_host:
        pipeline.set_roles(guest=guest, host=hosts)
    else:
        pipeline.set_roles(guest=guest, host=hosts[0])

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=hosts[0]).component_param(table=host_train_data)
    if is_multi_host:
        reader_0.get_party_instance(role='host', party_id=hosts[1]).component_param(table=host_train_data)

    # define DataTransform components
    data_transform_0 = DataTransform(name="data_transform_0")  # start component numbering at 0

    # get DataTransform party instance of guest
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(role='guest', party_id=guest)
    # configure DataTransform for guest
    data_transform_0_guest_party_instance.component_param(with_label=True, output_format="dense")
    # get and configure DataTransform party instance of host
    if host_dense_output:
        output_format = 'dense'
    else:
        output_format = 'sparse'
    if is_multi_host:
        data_transform_0.get_party_instance(role='host', party_id=hosts). \
            component_param(with_label=False,
                            output_format=output_format)
    else:
        data_transform_0.get_party_instance(role='host', party_id=hosts[0]). \
            component_param(with_label=False,
                            output_format=output_format)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")

    hetero_feature_binning_0 = HeteroFeatureBinning(**bin_param)

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    # set data input sources of intersection components
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    # set train & validate data of hetero_lr_0 component
    pipeline.add_component(hetero_feature_binning_0, data=Data(data=intersection_0.output.data))

    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    # pipeline.fit(work_mode=work_mode)
    return pipeline


def make_asymmetric_dsl(config, namespace, guest_param, host_param, dataset='breast', is_multi_host=False,
                        host_dense_output=True):
    parties = config.parties
    guest = parties.guest[0]
    hosts = parties.host

    if dataset == 'breast':
        guest_table_name = 'breast_hetero_guest'
        host_table_name = 'breast_hetero_host'
    elif dataset == 'default_credit':
        guest_table_name = 'default_credit_hetero_guest'
        host_table_name = 'default_credit_hetero_host'
    else:
        raise ValueError(f"dataset: {dataset} cannot be recognized")

    guest_train_data = {"name": guest_table_name, "namespace": f"experiment{namespace}"}
    host_train_data = {"name": host_table_name, "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    if is_multi_host:
        pipeline.set_roles(guest=guest, host=hosts)
    else:
        pipeline.set_roles(guest=guest, host=hosts[0])

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=hosts[0]).component_param(table=host_train_data)
    if is_multi_host:
        reader_0.get_party_instance(role='host', party_id=hosts[1]).component_param(table=host_train_data)

    # define DataTransform components
    data_transform_0 = DataTransform(name="data_transform_0")  # start component numbering at 0

    # get DataTransform party instance of guest
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(role='guest', party_id=guest)
    # configure DataTransform for guest
    data_transform_0_guest_party_instance.component_param(with_label=True, output_format="dense")
    # get and configure DataTransform party instance of host
    if host_dense_output:
        output_format = 'dense'
    else:
        output_format = 'sparse'
    if is_multi_host:
        data_transform_0.get_party_instance(role='host', party_id=hosts). \
            component_param(with_label=False,
                            output_format=output_format)
    else:
        data_transform_0.get_party_instance(role='host', party_id=hosts[0]). \
            component_param(with_label=False,
                            output_format=output_format)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")

    hetero_feature_binning_0 = HeteroFeatureBinning(name="hetero_feature_binning_0")
    hetero_feature_binning_0.get_party_instance(role='guest', party_id=guest).component_param(**guest_param)
    if is_multi_host:
        hetero_feature_binning_0.get_party_instance(role='host', party_id=hosts).component_param(**host_param)
    else:
        hetero_feature_binning_0.get_party_instance(role='host', party_id=hosts[0]).component_param(**host_param)

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    # set data input sources of intersection components
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    # set train & validate data of hetero_lr_0 component
    pipeline.add_component(hetero_feature_binning_0, data=Data(data=intersection_0.output.data))

    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    # pipeline.fit(work_mode=work_mode)
    return pipeline
