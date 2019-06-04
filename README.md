# FATE
FATE (Federated AI Technology Enabler) is an open-source project initiated by Webank's AI Department to provide a secure computing framework to support the federated AI ecosystem. It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). It supports federated learning architectures and secure computation of various machine learning algorithms, including logistic regression, tree-based algorithms, deep learning and transfer learning.


<https://www.fedai.org/>

## Install
FATE can be installed on Linux or Mac. Now, FATE can support standalone and cluster deployments.FATE can be installed on Linux by using:

```
git clone https://github.com/WeBankFinTech/FATE.git
```

#### Standalone
##### Docker version
```
FATE $ sh build_standalone_docker.sh
FATE $ CONTAINER_ID=`docker run -t -d  fate/standalone`
FATE $ docker exec -t -i ${CONTAINER_ID} bash
```

There are a few algorithms under `examples/` folder, try them out!

##### Manual version
```
FATE (venv) $ pip install -r requirements.txt
FATE (venv) $ export PYTHONPATH=`pwd -P`
```

#### Cluster
FATE also provides a distributed runtime architecture for Big Data scenario. Migration from standalone to cluster requires configuration change only. No algorithm change is needed. 

To deploy FATE on a cluster, please refer to cluster deploy guide at [`cluster-deploy`](https://github.com/WeBankFinTech/FATE/tree/master/cluster-deploy). 

## Running Tests

A script to run all the unittests has been provided in ./federatedml/test folder. 

Once FATE is installed, tests can be run using:

> sh ./federatedml/test/run_test.sh

All the unittests shall pass if FATE is installed properly. 

## Example Programs

###  Start Programs

Fate also provides several sample programs in the **examples** directory for each algorithm. You can check out for 
the detailed documentation in each specific algorithm directory. Here, we take homogeneous LR as an example. In **conf**
directory, we have provided several json file as configuration templates. The load_file.json is used for loading data into
distributed database. The three runtime_conf.json are configuration files for arbiter, guest and hosts respectively. 
Please check out the meaning of each parameter in out documentation. 

For quick start our program, we provide a standalone version which simulate three party in one machine. To start standalone 
version, please run the prepared shell file like:

> sh run_logistic_regression_standalone.sh

 Boom, a HomoLR program has been started up. This program will use the setting in configuration files. Please note that the 
 parameters in algorithm part (LogsiticParam for HomoLR, for example) are supposed to be identity among three parties.

 As for a real clustering version, we have also provided a quick start script which named **run_XXX_cluster.sh** (run_logistic_regression_cluster.sh for HomoLR)
 But this time, you need to specify the role, jobid and partyid for three parties. Then run the following three instructions in
 the corresponding party respectively:

 > sh run_logistic_regression_cluster.sh guest $jobid $guestpartyid $hostpartyid $arbiterpartyid 

 > sh run_logistic_regression_cluster.sh host $jobid $guestpartyid $hostpartyid $arbiterpartyid 

 > sh run_logistic_regression_cluster.sh arbiter $jobid $guestpartyid $hostpartyid $arbiterpartyid

### Check log files

The logs are provided in the **logs** directory. A sub-directory will be generated with the name of jobid. All the log files are
listed in this directory. The trained and validation result are shown in workflow.log. Feel free to check out each log file 
for more training details. 

## Getting Involved

*  Join us on google groups [Fate-FedAI Group](https://groups.google.com/forum/#!forum/fate-fedai). You can ask questions and participate in the development discussion.

*  For any frequently asked questions, you can check in [FAQ](https://github.com/WeBankFinTech/FATE/wiki).  

*  Please report bugs by submitting [issues](https://github.com/WeBankFinTech/FATE/issues). 

*  Submit contributions using [pull requests](https://github.com/WeBankFinTech/FATE/pulls)


## License
[Apache License 2.0](LICENSE)
