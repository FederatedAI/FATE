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
from pipeline.component.dataio import DataIO
from pipeline.component.evaluation import Evaluation
from pipeline.component.hetero_lr import HeteroLR
from pipeline.component.intersection import Intersection
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.interface.model import Model


def prettify(response, verbose=True):
    if verbose:
        print(json.dumps(response, indent=4, ensure_ascii=False))
        print()
    return response


def make_normal_dsl(config, namespace, lr_param, is_multi_host=False, has_validate=False,
                    host_dense_output=True, **kwargs):
    parties = config.parties
    guest = parties.guest[0]
    if is_multi_host:
        hosts = parties.host
    else:
        hosts = parties.host[0]
    arbiter = parties.arbiter[0]

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    guest_eval_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_eval_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    train_line = []
    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=hosts, arbiter=arbiter)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=hosts).algorithm_param(table=host_train_data)

    # define DataIO components
    dataio_0 = DataIO(name="dataio_0")  # start component numbering at 0

    # get DataIO party instance of guest
    dataio_0_guest_party_instance = dataio_0.get_party_instance(role='guest', party_id=guest)
    # configure DataIO for guest
    dataio_0_guest_party_instance.algorithm_param(with_label=True, output_format="dense")
    # get and configure DataIO party instance of host
    dataio_0.get_party_instance(role='host', party_id=hosts).algorithm_param(with_label=False)

    train_line.append(dataio_0)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")
    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))

    train_line.append(intersection_0)

    last_cpn = None
    if has_validate:
        reader_1 = Reader(name="reader_1")
        reader_1.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_eval_data)
        reader_1.get_party_instance(role='host', party_id=hosts).algorithm_param(table=host_eval_data)
        pipeline.add_component(reader_1)
        last_cpn = reader_1
        for cpn in train_line:
            cpn_name = cpn.name
            new_name = "_".join(cpn_name.split('_')[:-1] + ['1'])
            validate_cpn = type(cpn)(name=new_name)
            if hasattr(cpn.output, "model"):
                pipeline.add_component(validate_cpn, data=Data(data=last_cpn.output.data),
                                       model=Model(cpn.output.model))
            else:
                pipeline.add_component(validate_cpn, data=Data(data=last_cpn.output.data))
            last_cpn = validate_cpn

    hetero_lr_0 = HeteroLR(**lr_param)
    if has_validate:
        pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data,
                                                      validate_data=last_cpn.output.data))
    else:
        pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data))

    evaluation_data = [hetero_lr_0.output.data]
    if has_validate:
        hetero_lr_1 = HeteroLR(name='hetero_lr_1')
        pipeline.add_component(hetero_lr_1, data=Data(test_data=last_cpn.output.data),
                               model=Model(hetero_lr_0.output.model))
        evaluation_data.append(hetero_lr_1.output.data)

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")
    pipeline.add_component(evaluation_0, data=Data(data=evaluation_data))

    pipeline.compile()
    return pipeline
