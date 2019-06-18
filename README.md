# FATE
FATE (Federated AI Technology Enabler) is an open-source project initiated by Webank's AI Department to provide a secure computing framework to support the federated AI ecosystem. It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). It supports federated learning architectures and secure computation of various machine learning algorithms, including logistic regression, tree-based algorithms, deep learning and transfer learning.


<https://www.fedai.org/>

## Getting Involved

*  Join our maillist [Fate-FedAI Group IO](https://groups.io/g/Fate-FedAI). You can ask questions and participate in the development discussion.

*  For any frequently asked questions, you can check in [FAQ](https://github.com/WeBankFinTech/FATE/wiki).  

*  Please report bugs by submitting [issues](https://github.com/WeBankFinTech/FATE/issues). 

*  Submit contributions using [pull requests](https://github.com/WeBankFinTech/FATE/pulls)

## Install
FATE can be installed on Linux or Mac. Now, FATE can support standalone and cluster deployments.FATE can be installed on Linux by using:

```
git clone https://github.com/WeBankFinTech/FATE.git
```
Software environment :jdk1.8+、Python3.6、python virtualenv、mysql5.6+、redis-5.0.2

#### Standalone
##### Docker version
```
FATE $ sh build_standalone_docker.sh
FATE $ CONTAINER_ID=`docker run -t -d  fate/standalone`
FATE $ docker exec -t -i ${CONTAINER_ID} bash
```

There are a few algorithms under [examples/](./examples) folder, try them out!

##### Manual version
```
FATE (venv) $ pip install -r requirements.txt
FATE (venv) $ export PYTHONPATH=`pwd -P`
```

#### Cluster
FATE also provides a distributed runtime architecture for Big Data scenario. Migration from standalone to cluster requires configuration change only. No algorithm change is needed. 

To deploy FATE on a cluster, please refer to cluster deploy guide at [cluster-deploy](cluster-deploy). 

## Running Tests

A script to run all the unittests has been provided in ./federatedml/test folder. 

Once FATE is installed, tests can be run using:

> sh ./federatedml/test/run_test.sh

All the unittests shall pass if FATE is installed properly. 

## Example Programs

###  Start Programs

FATE also provides several sample programs in the [examples/](./examples) directory for each algorithm. You can check out for 
the detailed documentation in each specific algorithm directory. Here, we take heterogeneous LR as an example. In **conf** directory, we have provided several json file as configuration templates. The load_file.json is used for loading data into distributed database. The three runtime_conf.json are configuration files for arbiter, guest and hosts respectively.
Please check out the meaning of each parameter in out documentation. 

For quick start our program, we provide a standalone version which simulate three party in one machine. To start standalone version, please run the prepared shell file like:

> cd $FATE_install_path/examples/hetero_logistic_regression/

> sh run_logistic_regression_standalone.sh

 Boom, a HeteroLR program has been started up. This program will use the setting in configuration files. Please note that the parameters in algorithm part (LogsiticParam for HeteroLR, for example) are supposed to be identity among three parties.

### Run Cluster Version
In cluster version, you can use task-manager which is a tool help you start all the parties easily.
> cd $FATE_install_path/examples/hetero_logistic_regression/

#### load data
Before starting a cluster version task, you need to load data among all the data-providers. We have prepared some example data for hetero-lr. You should edit the file conf/load_file_tm_guest and conf/load_file_tm_host to make sure the data file path is right.
Then upload data using:

In role guest:
>  python $FATE_install_path/arch/task_manager/task_manager_client.py -f upload -c conf/load_file_tm_guest.json


In role host:
>  python $FATE_install_path/arch/task_manager/task_manager_client.py -f upload -c conf/load_file_tm_host.json


After load data, you can get "table_name" and "namespace" which have been edit in the configure file and please be attention to that the "table_name" should be different for each upload.

#### run task
Then, you need to edit a config file for role guest. A sample config file like *test_hetero_lr_workflow.json* has been provided in this folder. As the sample file shows, the parameters that are different among all the parties should be set in role_parameters respectively. On the other hand, those parameters that are same should be put in algorithm_parameters.


You should re-write the configure of  role guest "train_input_table" using "table_name" you have got in guest's load data, and "train_input_namespace" using "namespace" you have got. The same as the configure of  role host using "table_name" and "namespace" you got after host load data.

If you want to predict, please write the configure of "predict_input_table" and "predict_input_namespace". The configure of "model_table", "predict_output_table" and "evaluation_output_table" in role guest or host should be different for each task.


After finish editing, you can run the following command to start the task:

> python $FATE_install_path/arch/task_manager/task_manager_client.py -f workflow -c conf/test_hetero_lr_workflow.json

After running this command, a jobid will be generated automatically for you.

### Check log files

The logs are provided in the **logs** directory. A sub-directory will be generated with the name of jobid. All the log files are
listed in this directory. The trained and validation result are shown in workflow.log. Feel free to check out each log file 
for more training details. 

## License
[Apache License 2.0](LICENSE)
