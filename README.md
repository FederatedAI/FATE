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
FATE provides a standalone version of the docker for experience.please refer to docker version deploy guide at [docker-deploy](./standalone-deploy/docker).

##### Manual version

FATE provides a tar package with basic components to enable users to run FATE in a stand-alone environment, in which users are required to install dependent components on their own.please refer to manual deploy guide at [manual-deploy](./standalone-deploy/Manual). 

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

FATE also provides several sample programs in the [examples/](./examples) directory for each algorithm. Here, we take heterogeneous LR as an example.
In **dsl_example** directory, FATE provides heterogeneous LR running dsl in test_hetero_lr_job_dsl.json. Users only have to do the following several steps to start and heterogeneous LR example. For more details, please refer to the README in example folder.

#### Step1: Define upload data config file and upload data

To make FATE be able to use your data, you need to upload them. Thus, a upload-data conf is needed. A sample file named "upload_data.json" has been provided in **dsl_example** folder. For more detail information please check out [upload doc](./doc/upload_data_guide.md)

##### Field Specification
1. file: file path
2. head: Specify whether your data file include a header or not
3. partition: Specify how many partitions used to store the data
4. table_name & namespace: Indicators for stored data table.
##### Upload data
Before starting a task, you need to load data among all the data-providers. To do that, a load_file config is needed to be prepared.  Then run the following command:

> python ${your_install_path}/fate_flow/fate_flow_client.py -f upload -c dsl_examples/upload_data.json

After uploaded data, you can get "table_name" and "namespace" which have been edit in the configure file and
please be attention to that the "table_name" should be different for each upload.

##### Step2: Define configuration for each specific component and running the example.
This test_hetero_lr_job_conf.json config file is used to config parameters for all components among every party.
1. initiator: Specify the initiator's role and party id
2. role: Indicate all the party ids for all roles.
3. role_parameters: Those parameters are differ from roles and roles are defined here separately. Please note each parameter are list, each element of which corresponds to a party in this role.
4. algorithm_parameters: Those parameters are same among all parties are here.

To make things easily, FATE provides many default runtime parameters setting, users only need to modify the parties guest and host uploaded dataset' name & namespace.
After configuration,  execute the following command and a heterogeneous LR running example is started!
> python ${your_install_path}/fate_flow/fate_flow_client.py -f submitJob -d dsl_examples/test_hetero_lr_job_dsl.json -c dsl_example/test_hetero_lr_job_conf.json

For more details about DSL config and submit runtime conf, please refer to [this doc](./doc/dsl_conf_setting_guide.md)

### Check log files
Now you can check out the log in the following path: ${your_install_path}/logs/{your jobid}, All the log files are listed in this directory. Feel free to check out each log file for more training details.

## License
[Apache License 2.0](LICENSE)
