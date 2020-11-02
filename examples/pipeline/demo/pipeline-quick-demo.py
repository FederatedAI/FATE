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

import json
from pipeline.backend.config import Backend, WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component import Reader, DataIO, Intersection, HeteroSecureBoost, Evaluation
from pipeline.interface import Data
from pipeline.runtime.entity import JobParameters


guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
host_train_data = {"name": "breast_hetero_host", "namespace": "experiment"}

# initialize pipeline
pipeline = PipeLine().set_initiator(role="guest", party_id=9999).set_roles(guest=9999, host=10000)

# define components
reader_0 = Reader(name="reader_0")
reader_0.get_party_instance(role="guest", party_id=9999).component_param(table=guest_train_data)
reader_0.get_party_instance(role="host", party_id=10000).component_param(table=host_train_data)
dataio_0 = DataIO(name="dataio_0", with_label=True)
dataio_0.get_party_instance(role="host", party_id=10000).component_param(with_label=False)
intersect_0 = Intersection(name="intersection_0")
hetero_secureboost_0 = HeteroSecureBoost(name="hetero_secureboost_0",
                                         num_trees=5,
                                         bin_num=16,
                                         task_type="classification",
                                         objective_param={"objective": "cross_entropy"},
                                         encrypt_param={"method": "iterativeAffine"},
                                         tree_param={"max_depth": 3})
evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

# add components to pipeline, in order of task execution
pipeline.add_component(reader_0)\
    .add_component(dataio_0, data=Data(data=reader_0.output.data))\
    .add_component(intersect_0, data=Data(data=dataio_0.output.data))\
    .add_component(hetero_secureboost_0, data=Data(train_data=intersect_0.output.data))\
    .add_component(evaluation_0, data=Data(data=hetero_secureboost_0.output.data))

# compile & fit pipeline
pipeline.compile().fit(JobParameters(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE))

# query component summary
print(json.dumps(pipeline.get_component("evaluation_0").get_summary(), indent=4))
