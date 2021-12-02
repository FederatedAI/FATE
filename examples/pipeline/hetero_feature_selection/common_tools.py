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
from pipeline.component import DataStatistics
from pipeline.component import DataTransform
from pipeline.component import HeteroFastSecureBoost
from pipeline.component import HeteroFeatureBinning
from pipeline.component import HeteroFeatureSelection
from pipeline.component import HeteroSecureBoost
from pipeline.component import Intersection
from pipeline.component import PSI
from pipeline.component import Reader
from pipeline.component import FederatedSample
from pipeline.component import FeatureScale
from pipeline.interface import Data
from pipeline.interface import Model


def prettify(response, verbose=True):
    if verbose:
        print(json.dumps(response, indent=4, ensure_ascii=False))
        print()
    return response


def make_normal_dsl(config, namespace, selection_param, is_multi_host=False,
                    host_dense_output=True, **kwargs):
    parties = config.parties
    guest = parties.guest[0]
    if is_multi_host:
        hosts = parties.host
    else:
        hosts = parties.host[0]

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    guest_eval_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_eval_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=hosts)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=hosts).component_param(table=host_train_data)

    # define DataTransform components
    data_transform_0 = DataTransform(name="data_transform_0")  # start component numbering at 0

    # get DataTransform party instance of guest
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(role='guest', party_id=guest)
    # configure DataTransform for guest
    data_transform_0_guest_party_instance.component_param(with_label=True, output_format="dense")
    # get and configure DataTransform party instance of host
    data_transform_0.get_party_instance(role='host', party_id=hosts).component_param(with_label=False)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))

    last_cpn = intersection_0
    selection_include_model = []
    if 'binning_param' in kwargs:
        hetero_feature_binning_0 = HeteroFeatureBinning(**kwargs['binning_param'])
        pipeline.add_component(hetero_feature_binning_0, data=Data(data=last_cpn.output.data))
        selection_include_model.append(hetero_feature_binning_0)
        # last_cpn = hetero_feature_binning_0

    if 'statistic_param' in kwargs:
        # print(f"param: {kwargs['statistic_param']}, kwargs: {kwargs}")
        statistic_0 = DataStatistics(**kwargs['statistic_param'])
        pipeline.add_component(statistic_0, data=Data(data=last_cpn.output.data))
        # last_cpn = statistic_0
        selection_include_model.append(statistic_0)

    if 'psi_param' in kwargs:
        reader_1 = Reader(name="reader_1")
        reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=guest_eval_data)
        reader_1.get_party_instance(role='host', party_id=hosts).component_param(table=host_eval_data)
        data_transform_1 = DataTransform(name="data_transform_1")
        intersection_1 = Intersection(name="intersection_1")
        pipeline.add_component(reader_1)
        pipeline.add_component(
            data_transform_1, data=Data(
                data=reader_1.output.data), model=Model(
                data_transform_0.output.model))
        pipeline.add_component(intersection_1, data=Data(data=data_transform_1.output.data))

        psi_0 = PSI(**kwargs['psi_param'])
        pipeline.add_component(psi_0, data=Data(train_data=intersection_0.output.data,
                                                validate_data=intersection_1.output.data))
        # last_cpn = statistic_0
        selection_include_model.append(psi_0)

    if 'sbt_param' in kwargs:
        secureboost_0 = HeteroSecureBoost(**kwargs['sbt_param'])

        pipeline.add_component(secureboost_0, data=Data(train_data=intersection_0.output.data))
        selection_include_model.append(secureboost_0)

    if "fast_sbt_param" in kwargs:
        fast_sbt_0 = HeteroFastSecureBoost(**kwargs['fast_sbt_param'])
        pipeline.add_component(fast_sbt_0, data=Data(train_data=intersection_0.output.data))
        selection_include_model.append(fast_sbt_0)

    hetero_feature_selection_0 = HeteroFeatureSelection(**selection_param)

    pipeline.add_component(hetero_feature_selection_0, data=Data(data=intersection_0.output.data),
                           model=Model(isometric_model=[x.output.model for x in selection_include_model]))
    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()
    return pipeline


def make_single_predict_pipeline(config, namespace, selection_param, is_multi_host=False,
                                 **kwargs):
    parties = config.parties
    guest = parties.guest[0]
    if is_multi_host:
        hosts = parties.host
    else:
        hosts = parties.host[0]

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    guest_eval_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_eval_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=hosts)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=hosts).component_param(table=host_train_data)

    # define DataTransform components
    data_transform_0 = DataTransform(name="data_transform_0")  # start component numbering at 0

    # get DataTransform party instance of guest
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(role='guest', party_id=guest)
    # configure DataTransform for guest
    data_transform_0_guest_party_instance.component_param(with_label=True, output_format="dense")
    # get and configure DataTransform party instance of host
    data_transform_0.get_party_instance(role='host', party_id=hosts).component_param(with_label=False)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=guest_eval_data)
    reader_1.get_party_instance(role='host', party_id=hosts).component_param(table=host_eval_data)
    data_transform_1 = DataTransform(name="data_transform_1")
    intersection_1 = Intersection(name="intersection_1")

    pipeline.add_component(reader_1)
    pipeline.add_component(
        data_transform_1, data=Data(
            data=reader_1.output.data), model=Model(
            data_transform_0.output.model))
    pipeline.add_component(intersection_1, data=Data(data=data_transform_1.output.data))

    sample_0 = FederatedSample(name='sample_0', fractions=0.9)
    pipeline.add_component(sample_0, data=Data(data=intersection_0.output.data))

    if "binning_param" not in kwargs:
        raise ValueError("Binning_param is needed")

    hetero_feature_binning_0 = HeteroFeatureBinning(**kwargs['binning_param'])
    pipeline.add_component(hetero_feature_binning_0, data=Data(data=sample_0.output.data))

    hetero_feature_binning_1 = HeteroFeatureBinning(name='hetero_feature_binning_1')
    pipeline.add_component(hetero_feature_binning_1, data=Data(data=intersection_1.output.data),
                           model=Model(hetero_feature_binning_0.output.model))

    hetero_feature_selection_0 = HeteroFeatureSelection(**selection_param)
    pipeline.add_component(hetero_feature_selection_0, data=Data(data=hetero_feature_binning_0.output.data),
                           model=Model(isometric_model=[hetero_feature_binning_0.output.model]))

    hetero_feature_selection_1 = HeteroFeatureSelection(name='hetero_feature_selection_1')
    pipeline.add_component(hetero_feature_selection_1, data=Data(data=hetero_feature_binning_1.output.data),
                           model=Model(hetero_feature_selection_0.output.model))

    scale_0 = FeatureScale(name='scale_0')
    scale_1 = FeatureScale(name='scale_1')

    pipeline.add_component(scale_0, data=Data(data=hetero_feature_selection_0.output.data))
    pipeline.add_component(scale_1, data=Data(data=hetero_feature_selection_1.output.data),
                           model=Model(scale_0.output.model))
    pipeline.compile()
    return pipeline
