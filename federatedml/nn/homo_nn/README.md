# Homogeneous Neural Networks 

Neural networks are probably the most popular machine learning algorithms in recent years. FATE provides a federated homogeneous neural network implementation.
We simplified the federation process into three parties. Party A represents Guest，which acts as a task trigger.
Party B represents Host, which is almost the same with guest except that Host does not initiate task.
Party C serves as a coordinator to aggregate models from guest/hosts and broadcast aggregated model.
 
## 1. Basic Process

As its name suggested, in Homogeneous Neural Networks, the feature spaces of guest and hosts are identical.
An optional encryption mode for model is provided. 
By doing this, no party can get the private model of other parties. 

<div style="text-align:center", align=center>
<img src="./images/homo_nn.png" alt="samples" width="500" height="500" /><br/>
Figure 1： Federated Homo NN Principle</div> 

The Homo NN process is shown in Figure 1. Models of Party A and Party B have the same neural networks structure.
In each iteration, each party trains its model on its own data. After that, all parties upload their encrypted (with random mask) model parameters to arbiter. The arbiter aggregates these parameters to form a federated model parameter, which will then be distributed to all parties for updating their local models. Similar to traditional neural network, the training process will stop when the federated model converges or the whole training process reaches a predefined max-iteration threshold.

Please note that random numbers are carefully generated so that the random numbers of all parties add up an zero matrix and thus disappear automatically. For more detailed explanations, please refer to [Secure Analytics: Federated Learning and Secure Aggregation](https://inst.eecs.berkeley.edu/~cs261/fa18/scribe/10_15.pdf). Since there is no model transferred in plaintext, except for the owner of the model, no other party can obtain the real information of the model.

## 2. Features
 
1. supported layers:
    - Dense
    
       ```json
       {
            "layer": "Dense",
            "units": #int,
            "activation": null,
            "use_bias": true,
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "zeros",
            "kernel_regularizer": null,
            "bias_regularizer": null,
            "activity_regularizer": null,
            "kernel_constraint": null,
            "bias_constraint": null
       }
       ```
    - Droupout
    
      ```json
      {
            "rate": #float,
            "noise_shape": null,
            "seed": null
      }
      ```    
    other layers listed in [tf.keras.layers](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/layers) will be supported in near feature.

2. support all optimizer listed in [tf.keras.optimizers](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/optimizers):
    - [Adadelta](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/optimizers/Adadelta)

      ```json
      {
            "optimizer": "Adadelta",
            "learning_rate": 0.001,
            "rho": 0.95,
            "epsilon": 1e-07
      }
      ```
    - [Adagrad](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/optimizers/Adagrad)
      ```json
      {
            "optimizer": "Adagrad",
            "learning_rate": 0.001,
            "initial_accumulator_value": 0.1,
            "epsilon": 1e-07
      }
      ```
    
    - [Adam](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/optimizers/Adam)
      ```json
      {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "amsgrad": false,
            "epsilon": 1e-07
      }
      ```
      
    - [Ftrl](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/optimizers/Ftrl)
      ```json
      {
            "optimizer": "Ftrl",
            "learning_rate": 0.001,
            "learning_rate_power": -0.5,
            "initial_accumulator_value": 0.1,
            "l1_regularization_strength": 0.0,
            "l2_regularization_strength": 0.0,
            "l2_shrinkage_regularization_strength": 0.0
      }
      ```
      
    - [Nadam](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/optimizers/Nadam)
      ```json
      {
            "optimizer": "Nadam",
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-07
      }
      ```
      
    - [RMSprop](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/optimizers/RMSprop)
      ```json
      {
            "optimizer": "RMSprop",
            "learning_rate": 0.001,
            "pho": 0.9,
            "momentum": 0.0,
            "epsilon": 1e-07,
            "centered": false
      }
      ```
      
    - [SGD](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/optimizers/SGD)
      ```json
      {
            "optimizer": "SGD",
            "learning_rate": 0.001,
            "momentum": 0.0,
            "nesterov": false
      }
      ```   

3. support all losses listed in [tf.keras.losses](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/losses):
    - binary_crossentropy
    - categorical_crossentropy
    - categorical_hinge
    - cosine_similarity
    - hinge
    - kullback_leibler_divergence
    - logcosh
    - mean_absolute_error
    - mean_absolute_percentage_error
    - mean_squared_error
    - mean_squared_logarithmic_error
    - poisson
    - sparse_categorical_crossentropy
    - squared_hinge

4. support multi-host. In fact, for model security reasons, at least two host parties are required.

## 3. Configuration
For more information on task configuration, please refer to the [documentation](../../../examples/federatedml-1.x-examples/README.md) under example first. In this part we only talk about the parameter configuration.

### overview
Since all parties training Homogeneous Neural Networks have the same network structure, a common practice is to configure parameters under algorithm_parameters, which is shared across all parties. The basic structure is:
```json
{
      "config_type": "nn",
      "nn_define": [#layer1, #layer2, ...]
      "batch_size": -1,
      "optimizer": #optimizer,
      "early_stop": {
        "early_stop": #early_stop_type,
        "eps": 1e-4
      },
      "loss": #loss,
      "metrics": [#metrics1, #metrics2, ...],
      "max_iter": 10
}
```
Some detailed examples can be found in the example directory.

### nn_define
Each layer is represented as an object in json. Please refer to supported layers in Features part.

### optimizer
A json object is needed, please refer to supported optimizers in Features part.

### loss
A string is needed, please refer to supported losses in Features part.

### others
1. batch_size: a positive integer or -1 for full batch
2. max_iter:  max aggregation number, a positive integer,
3. early_stop: diff or abs
4. metrics: a string name, refer to [doc](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/metrics), such as Accuracy, AUC ...