## Quick Start

The steps of running neural network based logistic regression is exactly the same as running other logistic regression algorithms. Please refer to [start-programs](https://github.com/WeBankFinTech/FATE#start-programs) section for details. 

Since we are using neural network for extracting representative features, we need to specify hyperparameters for neural network model. For now, only one-hidden-layer autoencoder is supported (We will add other models in the near future and you are more than welcome to add your own models). Therefore, you need to specify hyperparameters for this model in **LocalModelParam** section in both **guest_runtime_conf.json** file and **host_runtime_conf.json** file. These two configuration files are located in **conf/** folder.

Following picture shows an example of parameters in **LocalModelParam** section.

<div style="text-align:center", align=center>
<img src="./images/neural_network_model_param.png" />
</div>

* **LocalModelParam** specifies hyperparameters for building local model such as Autoencoder.
    * *input_dim:*: the dimension of the original input samples
    * *encode_dim:*: the dimension of the encoded (or hidden) representation.
    * *learning_rate*: learning rate for the local model.
    
Note that the default values of these hyperparameters are catered to breast dataset. If you want to use other datasets, you should update these hyperparameters.