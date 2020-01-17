## Homo Neural Network Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Training Task.

1. single_layer:
    dsl: test_homo_nn_train_then_predict.json
    runtime_config : test_homo_dnn_single_layer.json
   
2. multi_layer:
    dsl: test_homo_nn_train_then_predict.json
    runtime_config: test_homo_dnn_multi_layer.json
   
3. multi_label and multi-host:
    dsl: test_homo_nn_train_then_predict.json
    runtime_config: test_homo_dnn_multi_label.json

    
Users can use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.

#### Use Keras Directly

1. build your model use keras:

```python
>>>from tensorflow.keras.models import Sequential
>>>from tensorflow.keras.layers import Dense
>>>model = Sequential()
>>>model.add(Dense(units=1, input_shape=(30, )))
```

2. generate nn_define from keras:

```python
>>>json = model.to_json()
>>>print(json)
```
paste this to nn_define field, and adjust config_type from "nn" to "keras"

3. or use temperate:
```python
>>>from string import Template
>>>temp = open("test_homo_nn_keras_temperate.json")
>>>rumtime_conf_json = Template(temp).safe_substitute(nn_define=json)
>>>with open(...) as f:
    f.write(rumtime_conf_json)
```

