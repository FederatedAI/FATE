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
from pipeline.backend.pipeline import PipeLine
from pipeline.component import Reader, DataTransform, Intersection, HeteroSecureBoost, Evaluation
from pipeline.interface import Data

# table name & namespace in data storage
# data should be uploaded before running modeling task
guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
host_train_data = {"name": "breast_hetero_host", "namespace": "experiment"}

# initialize pipeline
# Party ids are indicators of parties involved in federated learning. For standalone mode,
# arbitrary integers can be used as party id.
pipeline = PipeLine().set_initiator(role="guest", party_id=9999).set_roles(guest=9999, host=10000)

# define components

# Reader is a component to obtain the uploaded data. This component are very likely to be needed.
reader_0 = Reader(name="reader_0")
# By the following way, you can set different parameters for different party.
reader_0.get_party_instance(role="guest", party_id=9999).component_param(table=guest_train_data)
reader_0.get_party_instance(role="host", party_id=10000).component_param(table=host_train_data)

# Data transform provided some preprocessing to the raw data, including extract label, convert data format,
# filling missing value and so on. You may refer to the algorithm list doc for more details.
data_transform_0 = DataTransform(name="data_transform_0", with_label=True)
data_transform_0.get_party_instance(role="host", party_id=10000).component_param(with_label=False)

# Perform PSI for hetero-scenario.
intersect_0 = Intersection(name="intersection_0")

# Define a hetero-secureboost component. The following parameters will be set for all parties involved.
hetero_secureboost_0 = HeteroSecureBoost(name="hetero_secureboost_0",
                                         num_trees=5,
                                         bin_num=16,
                                         task_type="classification",
                                         objective_param={"objective": "cross_entropy"},
                                         encrypt_param={"method": "paillier"},
                                         tree_param={"max_depth": 3})

# To show the evaluation result, an "Evaluation" component is needed.
evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

# add components to pipeline, in order of task execution
# The components are connected by indicating upstream data output as their input.
# Typically, a feature engineering component will indicate input data as "data" while
# the modeling component will use "train_data". Please check out carefully of the difference
# between hetero_secureboost_0 input and other components below.
# Here we are just showing a simple example, for more details of other components, please check
# out the examples in "example/pipeline/{component you are interested in}
pipeline.add_component(reader_0)\
    .add_component(data_transform_0, data=Data(data=reader_0.output.data))\
    .add_component(intersect_0, data=Data(data=data_transform_0.output.data))\
    .add_component(hetero_secureboost_0, data=Data(train_data=intersect_0.output.data))\
    .add_component(evaluation_0, data=Data(data=hetero_secureboost_0.output.data))


# compile & fit pipeline
pipeline.compile().fit()

# query component summary
print(f"Evaluation summary:\n{json.dumps(pipeline.get_component('evaluation_0').get_summary(), indent=4)}")
