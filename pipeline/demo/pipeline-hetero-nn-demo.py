from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.input import Input
from pipeline.component.intersection import Intersection
from pipeline.component.hetero_nn import HeteroNN
from pipeline.interface.data import Data
from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode


guest = 9999
host = 10000
arbiter = 10002

guest_train_data = {"name": "breast_b", "namespace": "hetero"}
host_train_data = {"name": "breast_a", "namespace": "hetero"}

input_0 = Input(name="train_data")
print ("get input_0's init name {}".format(input_0.name))

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)
dataio_0 = DataIO(name="dataio_0")

dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, output_format="dense")
dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

intersect_0 = Intersection(name="intersection_0")
hetero_nn_0 = HeteroNN(name="hetero_nn_0",
                       bottom_nn_define={
                           "class_name": "Sequential",
                           "config": {"name": "sequential",
                                      "layers": [{"class_name": "Dense",
                                                  "config": {"name": "dense", "trainable": True, "batch_input_shape": [None, 1], "dtype": "float32", "units": 3, "activation": "relu", "use_bias": True, "kernel_initializer": {"class_name": "Constant", "config": {"value": 1, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": None, "bias_regularizer": None, "activity_regularizer": None, "kernel_constraint": None, "bias_constraint": None}}]},
                           "keras_version": "2.2.4-tf",
                           "backend": "tensorflow"
                       },
                       interactive_layer_define={
                           "class_name": "Sequential",
                           "config": {"name": "sequential_3",
                                      "layers": [{"class_name": "Dense",
                                                  "config": {"name": "dense_3", "trainable": True, "batch_input_shape": [None, 3], "dtype": "float32", "units": 2, "activation": "relu", "use_bias": True, "kernel_initializer": {"class_name": "Constant", "config": {"value": 1, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": None, "bias_regularizer": None, "activity_regularizer": None, "kernel_constraint": None, "bias_constraint": None}}]}, "keras_version": "2.2.4-tf", "backend": "tensorflow"},
                       top_nn_define={
                           "class_name": "Sequential",
                           "config": {"name": "sequential_2",
                                      "layers": [{"class_name": "Dense",
                                                  "config": {"name": "dense_2", "trainable": True, "batch_input_shape": [None, 2], "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": True, "kernel_initializer": {"class_name": "Constant", "config": {"value": 1, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": None, "bias_regularizer": None, "activity_regularizer": None, "kernel_constraint": None, "bias_constraint": None}}]}, "keras_version": "2.2.4-tf", "backend": "tensorflow"},
                       loss="binary_crossentropy",
                       early_stop="diff",
                       epochs=5
      )

print ("get input_0's name {}".format(input_0.name))
pipeline.add_component(dataio_0, data=Data(data=input_0.data))
pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data))
pipeline.add_component(hetero_nn_0, data=Data(train_data=intersect_0.output.data))

# pipeline.set_deploy_end_component([dataio_0])
# pipeline.deploy_component([dataio_0])

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE,
             feed_dict={input_0:
                           {"guest": {9999: guest_train_data},
                            "host": {
                              10000: host_train_data
                             }
                            }

                       })

print (pipeline.get_component("intersection_0").get_output_data())
print (pipeline.get_component("dataio_0").get_model_param())
print (pipeline.get_component("hetero_nn_0").get_model_param())
# pipeline.get_component("intersection_0").summary("intersect_count", "intersect_rate")


# predict

pipeline.predict(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE,
                              feed_dict={input_0:
                                             {"guest":
                                                  {9999: guest_train_data},
                                              "host": {
                                                  10000: host_train_data
                                              }
                                              }
                                         })

with open("output.pkl", "wb") as fout:
    fout.write(pipeline.dump())
