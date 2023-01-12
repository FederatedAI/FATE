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
from pipeline.components.fate import HeteroLR
from pipeline.components.fate import Reader
from pipeline.components.fate import FeatureScale
from pipeline.components.fate import Intersection
from pipeline.components.fate import Evaluation
from pipeline.pipeline import StandalonePipeline


pipeline = StandalonePipeline().set_scheduler_party_id(party_id=10001).set_roles(
        guest=9999, host=10000, arbiter=10001)
reader_0 = Reader(name="reader_0")
reader_0.guest.component_param(path="file:///Users/maguoqiang/mgq/FATE-2.0-alpha-with-flow/FATE/"
                                    "examples/data/breast_hetero_guest.csv",
                               format="csv",
                               id_name="id",
                               delimiter=",",
                               label_name="y",
                               label_type="float32",
                               dtype="float32")
reader_0.guest.conf.set("test_reader_guest", 2)

reader_0.hosts[0].component_param(path="file:///Users/maguoqiang/mgq/FATE-2.0-alpha-with-flow/FATE/"
                                            "examples/data/breast_hetero_host.csv",
                                       format="csv",
                                       id_name="id",
                                       delimiter=",",
                                       label_name=None,
                                       dtype="float32")

reader_0.hosts[[0, 1]].conf.set("test_reader_guest", 2)

intersection_0 = Intersection(name="intersection_0",
                              method="raw",
                              input_data=reader_0.outputs["output_data"])

intersection_1 = Intersection(name="intersection_1",
                              method="raw",
                              input_data=reader_0.outputs["output_data"])

feature_scale_0 = FeatureScale(name="feature_scale_0",
                               method="standard",
                               train_data=intersection_0.outputs["output_data"])

feature_scale_1 = FeatureScale(name="feature_scale_1",
                               test_data=intersection_1.outputs["output_data"],
                               input_model=feature_scale_0.outputs["output_model"])

lr_0 = HeteroLR(name="lr_0",
                train_data=feature_scale_0.outputs["train_output_data"],
                validate_data=feature_scale_1.outputs["test_output_data"],
                max_iter=1,
                learning_rate=0.01,
                batch_size=100)

lr_0.conf.set("backend", "gpu")
lr_1 = HeteroLR(name="lr_1",
                test_data=feature_scale_1.outputs["test_output_data"],
                input_model=lr_0.outputs["output_model"])

evaluation_0 = Evaluation(name="evaluation_0",
                          runtime_roles="guest",
                          input_data=lr_0.outputs["train_output_data"])

pipeline.add_task(reader_0)
pipeline.add_task(feature_scale_0)
pipeline.add_task(feature_scale_1)
pipeline.add_task(intersection_0)
pipeline.add_task(intersection_1)
pipeline.add_task(lr_0)
pipeline.add_task(evaluation_0)

# pipeline.add_task(lr_1)
pipeline.conf.set("task_parallelism", 1)
pipeline.compile()
print(pipeline.get_dag())
pipeline.fit()
print(pipeline.get_task_info("feature_scale_0").get_output_model())
print(pipeline.get_task_info("lr_0").get_output_model())
print(pipeline.get_task_info("lr_0").get_output_data())
print(pipeline.get_task_info("evaluation_0").get_output_metrics())
print(pipeline.deploy([intersection_0, feature_scale_0, lr_0]))


predict_pipeline = StandalonePipeline()
reader_1 = Reader(name="reader_1")
reader_1.guest.component_param(path="file:///Users/maguoqiang/mgq/FATE-2.0-alpha-with-flow/FATE/"
                                    "examples/data/breast_hetero_guest.csv",
                               format="csv",
                               id_name="id",
                               delimiter=",",
                               label_name="y",
                               label_type="float32",
                               dtype="float32")

reader_1.hosts[0].component_param(path="file:///Users/maguoqiang/mgq/FATE-2.0-alpha-with-flow/FATE/"
                                       "examples/data/breast_hetero_host.csv",
                                       format="csv",
                                       id_name="id",
                                       delimiter=",",
                                       label_name=None,
                                       dtype="float32")


deployed_pipeline = pipeline.get_deployed_pipeline()
deployed_pipeline.intersection_0.input_data = reader_1.outputs["output_data"]

predict_pipeline.add_task(deployed_pipeline)
predict_pipeline.add_task(reader_1)

print("\n\n\n")
print(predict_pipeline.compile().get_dag())
predict_pipeline.predict()
