## Release 1.9.0
### Major Features and Improvements
> FederatedML
* Add elliptic curve based PSI algorithm, which allows 128-bit secure-level intersection of billion samples, 20x faster than RSA protocol under the same security level
* Support accurate intersection cardinality calculation
* Support for multi-column ID data; user may specify id column for PSI intersection and subsequent modeling usage
* Hetero NN supports torch backend and supports complex layers such as LSTM
* Add CoAE label reinforcement mechanism for vertical federated neural network
* Hetero NN supports multi-host modeling scenarios
* HeteroSecureBoost supports merging sub-models from all parties and exporting the merged model into lightgbm or PMML format
* HeteroLR and HeteroSSHELR support merging sub-models from all parties and exporting the merged model into sklearn or PMML format
* HeteroFeatureSelection supports anonymous feature selection
* Label Encoder adds automatic label type inference
* 10x faster local VIF computation in HeteroPearson, with added support for computing local VIF on linearly dependent columns
* Optimized feature engineering column processing logic
* HeteroFeatureBinning supports calculation of IV and WOE values during prediction
* Renewed feature anonymous generation logic

> FATE-ARCH
* Support python3.8+
* Support Spark 3x
* Renewed Federation module, RabbitMQ and Pulsar support client transmission mode
* Support Standalone, Spark, EggRoll heterogeneous computing engine interconnection

> Fate-Client
* PipeLine adds timeout retry mechanism
* Pipeline's `get_output_data` API now may return component output data in DataFrame-typed format


## Release 1.8.0
### Major Features and Improvements
> FederatedML
* Add non-coordinated-version Hetero Linear Regression, based on integrated Hetero GLM framework, with mixed protocol of HE and SPDZ
* Homo LR support one-vs-rest
* Add SecureBoost-MO algorithm to speed up multi-class classification of Hetero & Homo SecureBoost, 1.5x-5x faster
* Optimize Hetero SecureBoost Predict Transmission Data Size，reduce 75% bandwidth consumption if tree's max depth is small
* Speed up DH Intersection implementation, 30%+ faster
* Optimized Quantile Binning gk-summary structure & split point query，20%+ faster, less memory cost
* Support weighted training in non-coordinated Hetero Logistic Regression & Linear Regression
* Merge Hetero FastSecureBoost into Hetero SecureBoost as a boosting strategy option

> Fate-ARCH
* Adjustable task_cores for standalone FATE
* Enable Eggroll option to make computing output "IN_MEMORY" by default

> Fate-Test
* Include Paillier encryption performance evaluation
* Include SPDZ performance evaluation
* Optimized testsuite printout
* Include examples data upload and mnist download
* Provide pipeline to dsl convert tools

> Bug-Fix
* Fix bug for SPDZ when using default q_filed
* Fix multiple get problem of SPDZ
* Fix bugs of recursive-query homo feature binning
* Fix homo_nn's model aggregation problem
* Fix bug for hetero feature selection when using federated filter but some party's feature is empty.


## Release 1.7.2
### Major Features and Improvements
> FederatedML
* New batch strategy in coordinated Hetero LR: support masked batch data and batch shuffle
* Model inference protection enhancement for Hetero SecureBoost with FED-EINI algorithm
* Hetero SecureBoost supports split feature importance on host side, disables gain feature importance
* Offline SBT Feature transform component 

> Bug-Fix
* Fixed Bug for HeteroPearson with changing default q_field value for spdz
* Fix Data Transform's schema label name setting problem when `with_label` is False
* Add testing examples for new algorithm features, and delete deprecated params in algorithm examples.

> FATE-ARCH
* Support the loading of custom password encryption modules through plug-ins
* Separate the base connection address of the data storage table from the data table information, and compatible with historical versions


## Release 1.7.1.1
### Major Features and Improvements
> Deploy
* upgrade mysql to version 8.0.28

> Eggroll
* Support Eggroll v2.4.3, upgrade com.h2database:h2 to version 2.1.210, com.google.protobuf:protobuf-java to version 3.16.1

## Release 1.7.1
### Major Features and Improvements
> FederatedML
* Iterative Affine is disabled
* Speed up Hetero Feature Selection, 100x+ faster when feature dimension is high
* Speed up OneHot, 60x+ faster when feature dimension is high
* Data Statistics supports missing value, with improved efficiency
* Fix bug of quantile binning: may lose data when partitions hold too many instances
* Fix reconstruction reuse problem of SPDZ
* Fix Host's ineffective decay rate of Homo Logistic Regression
* Improved strategy for handling missing values when converting Homo SecureBoost using homo model convertor
* Improved presentation of Evaluation's confusing matrix

> FATE-Client
* Add Source Provider attribute to Pipeline components

> Eggroll
* Support Eggroll v2.4.2, fixed Log4j security bug


## Release 1.7.0
### Major Features and Improvements
> FATE-ARCH

* Support EggRoll 2.4.0 
* Support Spark-Local Computing Engine
* Support Hive Storage
* Support LocalFS Storage for Spark-Local Computing Engine 
* Optimizing the API interface for Storage session and table
* Simplified the API interface for Session, remove backend and workmode parameters
* Heterogeneous Engine Support: Federation between Spark-Local and Spark-Cluster
* Computing Engine, Storage Engine, Federation Engine are set in conf/service_conf.yaml when FATE is deployed
  
> FederatedML

* Optimized Hetero-SecureBoost: with gradient packing、cipher compressing, and sparse point statistics optimization, 4x+ faster
* Homo-SecureBoost supports memory-based histogram computation for more efficient tree building, 5x+ faster
* Optimized RSA Intersect with CRT optimization, 3x+ faster
* New two-party Hetero Logistic Regression Algorithm: mixed protocol of HE and MPC, without a trusted third party
* Support data with match-id,  separating match id and sample id
* New DH Intersect based on PH Key-exchange protocol 
* Intersect support cardinality estimation 
* Intersect adds optionally preprocessing step
* RSA and DH Intersect support cache
* New Feature Imputation module: can apply arbitrary imputation method to each column
* New Label Transform module: transform categorical label values
* Homo-LR, Homo-SecureBoost, Homo-NN now can convert models into sklearn、lightgbm、torch & tf-keras framework
* Hetero Feature Binning supports multi-class task, higher efficiency with label packing
* Hetero Feature Selection support multi-class iv filter
* Secure Information Retrieval supports multi-column retrieval
* Major training algorithms support warm-start and checkpoint : Homo & Hetero LR, Homo & Hetero-SecureBoost, Homo & Hetero NN
* Optimized Pailler addition operation, several times faster, Hetero-SecureBoost with Paillier speed up 2x+

> Fate-Client

* Pipeline supports uploading match id functionality
* Pipeline supports homo model conversion
* Pipeline supports model push to FATE-Serving
* Pipeline supports running jobs with specified FATE version

> FATE-Test

* Integrate FederatedML unittest
* Support for uploading image data
* Big data generation using storage interface, optimized generation logic
* Support for historical data comparison
* cache_deps and model_loader_deps support
* Run DSL Testsuite with specified FATE version


       
## Release 1.6.1
### Major Features and Improvements
> FederatedML
* Support single party prediction
* SIR support non-ascii id
* Selection support local iv filter
* Adjustable Paillier key length for Hetero LR
* Binning support iv calculation on categorical features
* Hetero LR one vs rest support evaluation during training

> FATE-Flow:
* Support mysql storage engine;
* Added service registry interface;
* Added service query interface;
* Support fate on WeDataSphere mode
* Add lock when writing `model_local_cache`
* Register the model download urls to zookeeper

> Bug-Fix:
* Fix error for deploying module with lack of partial upstream modules in multi-input cases
* Fix error for deploying module with multiple output, like data-statistics
* Fix job id length no more than 25 limitation
* Fix error when loss function of Hetero SecureBoost set to log-cosh
* Fix setting predict label to string-type error when Hetero SecureBoost predicts
* Fix error for HeteroLR without intercept
* Fix quantile error of Data Statistics with specified columns
* Fix string parsing error of OneHot with specified columns
* Some parameters can now take 0 or 1 integer values when valid range is [0, 1]


## Release 1.6.0
### Major Features and Improvements

> FederatedML

* Hetero SecureBoost: more efficient computation with GOSS, histogram subtraction, cipher compression, 2-4x faster
* Hetero GLM: improved communication efficiency, adjustable floating point precision, 2x faster 
* Hetero NN: adjustable floating point precision, support SelectiveBackPropagation and dropOut on interaction layer, 2x faster
* Hetero Feature Binning: improved algorithm with cipher compression, 2x faster
* Intersect: add split calculation option and adjustable random base fraction, 30% faster 
* Homo NN: restructure torch backend and enhanced grammar; train and predict with raw image data
* Intersect supports SM3 hashing method 
* Hetero SecureBoost: L1 penalty & adjustable min_child_weight to prevent overfitting 
* NEW SecureBoost Transformer: feature engineering module that encodes instances with leaf nodes from SecureBoost model
* Hetero Pearson: support local VIF computation 
* Hetero Feature Selection: support selection based on VIF and Pearson 
* NEW Homo Feature Binning: support virtual/recursive binning strategy
* NEW Sample Weight: set sample weights based on label or from feature column, Hetero GLM & Hetero SecureBoost support weighted training
* NEW Data Transformer: case-insensitive on data schema
* Local Baseline supports prediction task
* Cross Validation: output fold split history 
* Evaluation: add multi-result-unfold option which unfolds multi-classification evaluation result to several binary evaluation results in a one-vs-rest manner 

>System Architecture

* Added local file system directory path virtual storage engine to support image input data
* Added the message queue Pulsar cross-site transmission engine, which can be used with the Spark computing engine, and can be added to the Exchange role to support the star networking mode

> FATE-Test

* Add Benchmark performance for efficiency comparison; add mock data generation tool; support metrics comparison between training and validation sets
* FATE-Flow unittest for REST/CLI/SDK API and training-prediction workflow 

## Release 1.5.2
### Major Features and Improvements

> FederatedML
* SIR support non-ascii id
* Selection support local iv filter
* Adjustable Paillier key length for Hetero LR
* Binning support iv calculation on categorical features
* Hetero LR one vs rest support evaluation during training

> Fate-Flow
* Read data from mysql with ‘table bind’ command to map source table to FATE table
* FATE cluster push model for one-to-multiple FATE Serving clusters in one party

> System Architecture
* More efficient ‘sample’ api 

> Bug Fixes
* Fix error for deploying module with lack of partial upstream modules in multi-input cases
* Fix job id length no more than 25 limitation
* Fix error when loss function of Hetero SecureBoost set to log-cosh
* Fix setting predict label to string-type error when Hetero SecureBoost predicts
* Fix error for HeteroLR without intercept
* Fix torch import error
* Fix quantile error of Data Statistics with specified columns
* Fix string parsing error of OneHot with specified columns
* Some parameters can now take 0 or 1 integer values when valid range is [0, 1]


## Release 1.5.1
### Major Features and Improvements

> FederatedML
* Add Feldman Verifiable Secret Sharing protocol (contributed)
* Add Feldman Verifiable Sum Module (contributed)
* Updated FATE-Client and FATE-Test for new FATE-Flow
* Upgraded early stopping strategy: record best model for each metric

> Fate-Flow
* Optimize the model center, reconstruct publishing model, support deploy, load, bind, migrate operations, and add new interfaces such as model info
* Improve identity authentication and resource authorization, support party identity verification, and participate in the authorization of roles and components
* Optimize and fix resource manager, add task_cores job parameters to adapt to different computing engines

> Eggroll
* In one-way communication mode, add party identity authentication function, which needs to be used with FATE-Cloud

> Deploy
* Support 1.5.0 retain data upgrade to 1.5.1

### Bug Fixes
* Fix predict-cache in SecureBoost validation
* Fix job clean CLI

## Release 1.5.0（LTS）
### Major Features and Improvements

> FederatedML

* Refactored Hetero FTL with optional communication-efficiency mechanism, with 4x time efficiency improvement
* Hetero SecureBoost supports complete secure mode
* Hetero SecureBoost now can reduce time consumption over highly sparse data by using sparse matrix 
    computation on histogram aggregations.
* Hetero SecureBoost optimization: the communication round in prediction is reduced to no larger than tree depth, 
                                                         prediction speed is improved by 32 times in a 100-tree model.
* Addition of Hetero FastSecureBoost module, whose mixed/layered modeling method makes it twice as efficient as SecureBoost  
* Improved Hetero Federated Binning with 30%~50% time efficiency improvement
* Better GLM: >10% improvement in time efficiency
* FATE first unsupervised learning algorithm: Hetero KMeans
* Upgraded Hetero Feature Selection: add PSI filter and SecureBoost feature importance filter
* Add Data Split module: splitting data into train, validate, and test sets inside FATE modeling workflow
* Add DataStatistic module: compute min/max, mean, median, skewness, kurtosis, coefficient of variance, percentile, etc.
* Add PSI module for computing population stability index
* Add Homo OneHot module for one-hot encoding in homogeneous scenario
* Evaluation module adds metrics for clustering
* Optional FedProx mechanism for Homo LR, useful for training with non-iid data
* Add Oblivious Transfer Protocol and OT-based module Secure Information Retrieval
* Random Iterative Affine protocol, providing additional security

> Fate-Flow

* Brand new scheduling framework based on global state and optimistic concurrency control and support multiple scheduler
* Upgraded task scheduling: multi-model output for component, executing component in parallel, component rerun
* Add new DSL v2 which significantly improves user experiences in comparison to DSL v1. Several syntax error detection functions are supported in v2. Now DSL v1 and v2 are 
   compatible in the current FATE version
* Enhanced resource scheduling: remove limit on job number, base on cores, memory and working node according to different computing engine supports
* Add model registry, supports model query, import/export, model transfer between clusters
* Add Reader component: automatically dump input data to FATE-compatible format and cluster storage engine; now data from HDFS
* Refactor submit job configuration's parameters setting, support different parties use different job parameters when using dsl V2.

> System Architecture

* New architectural framework that supports a combination of different computing, storage, and transfer engines
* Support new engine combination: Spark、HDFS、RabbitMQ
* New data table management, standardized API for all different storage engines
* Rearrange FATE code structure, conf setting at one place, streamlined user experiment
* Support one-way network communication between parties, only one party needs to open the entrance network strategy

> FATE-Client

* Pipeline, a tool with a keras-like user interface and integrates TensorFlow, PyTorch, Keras in the backend, is used for fast federated model building with FATE
* Brand new CLI v2 with easy independent installation, user-friendly programming syntax & command-line prompt
* Support FLOW python language SDK
* Support PyPI

> FATE-Test

* Testsuite: For Fate function regressions
* Benchmark tool and examples for comparing modeling quality; provided examples include common models such as heterogeneous LR, SecureBoost, and NN
* Performance Statistics: Log now includes statistics on timing, API usage, and variable transfer


## Release 1.4.5
### Major Features and Improvements
> EggRoll
* RollSite supports the communication certificates

## Release 1.4.4
### Major Features and Improvements
> FATE-Flow
* Task Executor supports monkey patch
* Add forward API

## Release 1.4.3
### Major Features and Improvements
> FederatedML
* Fix bug of Hetero SecureBoost of sending tree weight info from guest to host.


## Release 1.4.2
### Major Features and Improvements
> FederatedML
* Optimize performance of Pearson which increases efficiency by more than twice.
* Optimize Min-test module: Add secure-boost as optional test task. Set partyid and work_mode as input parameters. Use pre-import data set as input so that improved test process.
* Support tok_k iv filter in feature selection module.
* Support filling missing value for tag:value format data in DataIO.
* Fix bug of lacking one layer of depth of tree in HeteroSecureBoost and support automatically alignment header of input data in predict process
* Standardize the naming of example data set and add a data pre-import script. 

> FATE-Flow
* Distinguish between user stop job and system stop job;
* Optimized some logs;
* Optimize zookeeper configuration
* The model supports persistent storage to mysql
* Push the model to the online service to support the specified storage address (local file and FATEFlowServer interface)


## Release 1.4.1
### Major Features and Improvements
> FederatedML
* Reconstructed Evaluation Module improves efficiency by 60 times
* Add PSI, confusion matrix, f1-score  and quantile threshold support for Precision/Recall in Evaluation.
* Add option to retain duplicated keys in Union.
* Support filter feature based on mode
* Manual filter allows manually set columns to retain
* Auto recoginize whether a data set includes a label column in predict process
* Bug-fix: Missing schema after merge in Union; Fail to align label of multi-class in homo_nn with PyTorch backend; Floating-point precision error and value error due to int-type input in Feature Scale

> FATE-Flow
* Allow the host to stop the job
* Optimize the task queue
* Automatically align the input table partitions of all participants when the job is running
* Fate flow client large file upload optimization
* Fixed some bugs with abnormal status


## Release 1.4.0
### Major Features and Improvements
> FederatedML
* Support Homo Secureboost
* Support AIC/BIC-based Stepwise for Linear Models
* Add Hetero Optimal Feature Binning, support iv/gini/chi-square/ks,and allow asymmetric binning methods 
* Interoperate with AI ecosystem: Add pytorch backend for Homo NN
* Homo Framework factorization, simplify developing homo algorithms
* Early stopping strategy for hetero algorithms.
* Local Baseline supports multi-class classification
* Add consistency check to Predict function
* Optimize validation strategy，3x speed up in-training validation

> FATE-Flow
* Refactoring model management, native file directory storage, storage structure is more flexible, more information
* Support model import and export, store and restore with reliable distributed system(Redis is currently supported)
* Using MySQL instead of Redis to implement Job Queue, reducing system complexity
* Support for uploading client local files
* Automatically detects the existence of the table and provides the destroy option
* Separate system, algorithm, scheduling command log, scheduling command log can be independently audited

> Eggroll  
>> Stability Boosts:
* New resource management components introduce the brand new session mechanism. Processors can be cleaned up with a simple method call, even the session goes wrong.
* Removes storage service. No C++ / native library compilation is needed.
* Federated learning algorithms can still work at a 28% packet loss rate.
>> Performance Boosts:
* Performances of federated learning algorithms are improved on Eggroll 2. Some algorithms get 10x performance boost.
* Join interface is 16x faster than pyspark under federated learning scenarios.
>> User Experiences Boosts:
* Quick deployment. Maven, pip, config and start.
* Light dependencies. Check our requirements.txt / pom.xml and see.
* Easy debugging. Necessary running contexts are provided. Runtime status are kept in log files and databases.
* Few daemon processes. And they are all JVM applications.


## Release 1.3.1
### Major Features and Improvements
>Deploy
* Support deploying by MacOS
* Support using external db
* Deploy JDK and Python environments on demand
* Improve MySQL and FATE Flow service.sh
* Support more custom deployment configurations in the default_configurations.sh, such as ssh_port, mysql_port and so one.

## Release 1.3.0
### Major Features and Improvements
>FederatedREC
* Add federated recommendation submodule
* Add heterogeneous Factoraization Machine
* Add hemogeneous Factoraization Machine
* Add heterogeneous Matrix Factorization
* Add heterogeneous Singular Value Decomposition
* Add heterogeneous SVD++ (Factorization Meets the Neighborhood)
* Add heterogeneous Generalized Matrix Factorization

>FederatedML
* Support Sparse data training in heterogeneous General Linear Model(Hetero-LR、Hetero-LinR、Hetero-PoissonR)
* Fix 32M limitation of quantile binning to support higher feature dimension
* Fix 32M limitation of histogram statistics for  SecureBoost to support higher feature dimension training.
* Add abnormal parameters and input data detection in OneHot Encoder
* fix not passing validate data to fit process to support evaluate validation data during training process

>FATE-Flow
* Add clean job CLI for cleaning output and intermediate results, including data, metrics and sessions
* Support for obtaining table namespace and name of output data via CLI
* Fix KillJob unsuccessful execution in some special cases
* Improve log system, add more exception and run time status prompts


## Release 1.2.0
### Major Features and Improvements
FederatedML
* Add heterogeneous Deep Neural Network
* Add Secret-Sharing Protocol-SPDZ
* Add heterogeneous feature correlation algorithm with SPDZ and support heterogeneous Pearson Correlation Calculation 
* Add heterogeneous SQN optimizer, available for Hetero-LogisticRegression and Hetero-LinearRegression, which can reduce communication costs significantly 
* Supports intersection for expanding duplicate IDs
* Support multi-host in heterogeneous feature binning
* Support multi-host in heterogeneous feature selection
* Support IV calculation for categorical features in heterogeneous feature binning
* Support transform raw feature value to WOE in heterogeneous feature binning 
* Add manual filters in heterogeneous feature selection
* Support performance comparison with sklearn's logistic regression
* Automatic object/table clean in training iteration procedure in Federation
* Improve transfer performance for large object
* Add automatic scripts for submitting and running tasks

FATE-Flow
* Add data management module for recording the uploaded data tables and the outputs of the model in the job running, and for querying and cleaning up CLI. 
* Support registration center for simplifying communication configuration between FATEFlow and FATEServing
* Restruct model release logic, FATE_Flow pushes model directly to FATE-Serving. Decouple FATE-Serving and Eggroll, and the offline and online architectures are connected only by FATE-Flow.
* Provide CLI to query data upload record
* Upload and download data support progress statistics by line
* Add some abnormal diagnosis tips
* Support adding note information to job

>Native Deploy
* Fix bugs in EggRoll startup script, add mysql, redis startup options.
* Disable host name resolution configuration for mysql service.
* The version number of each module of the software packaging script is updated using the automatic acquisition mode.



## Release 1.1.1
### Major Features and Improvements
* Add cluster deployment support based on ubuntu operating system。
* Add union component which support data merging. 
* Support indicating partial columns in Onehot Encoder
* Support intermediate data cleanup after the task ends
* Accelerated Intersection
* Optimizing the deployment process


### Bug Fixes
* Fix a bug of secureboost' early stop 
* Fix a bug in download api
* Fix bugs of spark-backend


## Release 1.1
### Major Features and Improvements
>FederatedML
* Provide a general algorithm framework for homogeneous federated learning, which supports Secure Aggregation 
* Add homogeneous Deep Neural Network
* Add heterogeneous Linear Regression
* Add heterogeneous Poisson Regression
* Support multi-host in heterogeneous Logistic Regression
* Support multi-host in heterogeneous Linear Regression
* Support multi-host Intersection
* Accelerated Intersection by usage of cache
* Reconstruct heterogeneous Generalized Linear Models Framework
* Support affine homomorphic encryption in heterogeneous SecureBoost
* Support input data with missing value in heterogeneous SecureBoost
* Support evaluation during training on both train and validate data
* Add spark as computing engine

>FATE-Flow
* Upload and Download support CLI for querying job status
* Support for canceling waiting job
* Support for setting job timeout
* Support for storing a job scheduling log in the job log folder
* Add authentication control Beta version, including component, command, role

## Release 1.0.2
### Major Features and Improvements
* Python and JDK environment are required only for running standalone version quick experiment
* Support cluster version docker deployment
* Add deployment guide in Chinese
* Standalone version job for quick experiment is supported when cluster version deployed. 
* Python service log will remain for 14 days now.


### Bug Fixes
* Fix bugs of multi-host support in Cross-Validation
* Fix bugs of showing up evaluation metrics when both train and eval exist
* Add links for each algorithm module in FederatedML home page README


## Release 1.0.1
### Bug Fixes
* Fix bugs for evaluation data type 
* Fix bugs for feature binning to take abnormal values into consideration
* Fix bugs for train and eval
* Fix bugs in binning merge
* Fix bugs in Samplers
* Fix federated feature selection feature filter bug
* Support upload file  in version argument
* Support get serviceRoleName from configuration


## Release 1.0
### Major Features and Improvements
>This version includes two new products of FATE, FATE-Board, and FATE-Flow respectively, FATE-Board as a visual tool for federation modeling, and FATE-Flow is an end to end pipeline platform for federated learning. This version contains important improvements to the FederatedML, which better tracks the running progress of federated learning algorithms.

>FATE-Board
* Federated Learning Job DashBoard
* Federated Learning Job Visualisation
* Federated Learning Job Management
* Real-time Log Panel


>FATE-FLOW
 * DAG defines Pipeline
 * Federated Multi-party asymmetric DSL parser
 * Federated Learning lifecycle management
 * Federated Task collaborative scheduling
 * Tracking for data, metric, model and so on
 * Federated Multi-party model management
 
>FederatedML
* Update all algorithm modules running mechanism for supporting federated modeling pipeline by FATE-Flow
* Intermediate statistic result callback is available and visualizable in FATE-Board for all algorithm modules.
* Support Nesterov Momentum SGD Optimizer
* Add Homomorphic Encryption Scheme Based on Affine Transforms
* Support sparse input-format in federated feature binning
* Update evaluation metrics, such as ks, roc, gain, lift curve and so on 
* Update algorithm's parameter-define class

>FATE-Serving
* Add online federated modeling pipeline DSL parser for online federated inference


## Release 0.3.2
### Bug Fixes
* Adjust the Logic of Online Service Module
* Adjust the log format
* Replace the grpc connection pool of the online service module
* Improving Model Processing Details


## Release 0.3.1
### Bug Fixes
* fix feature scale bugs in v0.3
* fix federated feature selection bugs in v0.3

## Release 0.3

### Major Features and Improvements

> FederatedML
* Support OneVsALL for multi-label classification task
* Add trash-recycle in Hetero Logistic Regression
* Add numeric stable for sigmoid and log_logistic function.
* Support different calculation mode in Hetero Logistic Regression and Hetero SecureBoost
* Decouple Federated Feature Binning and Federated Feature Selection
* Add feature importance calculation in Hetero SecureBoost
* Add multi-host in Hetero SecureBoost
* Support tag:value sparse format input data 
* Support output intersect-id with feature-instance in Intersection
* Support OneHot encoding module.
* Support bucket binning for Federated Feature Binning.
* Support add, sub, mul, div ,gt, lt ,eq, etc mathematical operator on Fixed-Point data
* Add authority validation for parameter setting

> FATE-Serving
* Add multi-level cache for multi-party inference result
* Add startInferceJob and getInferenceResult interfaces to support the inference process asynchronization
* Normalized inference return code
* Real-time logging of inference summary logs and inferential detail logs
* Improve the loading of the pre and post processing adapter and data access adapter for host

> EggRoll
* New computing and storage APIs
* Stability optimizations
* Performance optimizations
* Storage usage improvements

> Example
* Add Mini-FederatedML test task example
* Using task manager to submit distributed task for current examples  

### Bug Fixes and Other Changes
* fix detect onehot max column overflow bug.
* fix dataio dense format not reading host data header bug.
* fix bugs of call of statistics function
* fix bug for federated feature selection that at least one feature remains for each party
* Not allowing so small batch size in LR module for safety consideration.
* fix naming error in federated feature selection module.
* Fix the bug of automated publishing model information in some extreme cases
* Fixed some overflow bugs in fixed-point data
* fix many other bugs.


## Release 0.2

### Major Features and Improvements
> WorkFlow
* Add Model PipleLine
* Add Hetero Federated Feature Binning workflow
* Add Hetero Federated Feature Selection workflow
* Add hetero dnn workflow
* Add intersection operator before train, predict and cross_validation

> FederatedML
* Support svm-light sparse format inputdata 
* Support tag sparse format inputdata
* Add Hetero Federated Feature Binning
* Add Hetero Federated Feature Selection
* Add Feature Scaler: MinMaxScaler & StandardScaler
* Add Feature Imputer for missing value filling
* Add Data Statistic for datainstance
* Support encoding and main calculation role configurable for RAW Intesection 
* Add Sampler: RandomSampler & StratifiedSampler 
* Support regression in SecureBoost
* Support regression evaluation   
* Support Decentralized FTL
* Add feature extracting by DNN
* Change Model Format to ProtoBuf
* Add abnormal parameter detection
* Add abnormal input data detection


> FATE-Serving(An online inference for federated learning models)	
*	Dynamic Loading Federated Learning Models.
*	Real-time Prediction Using Federated Learning Models.

> Model Management
*	Versioning
*	Reproducibility
*	Queries, Search

> Task Manager
*  Add Load File/ Download File
*  Add Import ID from Local File
*  Add Start workflow
*  Add workflow Job Queue
*  Add Query Job Status
*  Add Get Runtime conf
*  Add Delete Task

> EggRoll
*  Add Node Manager for multiprocessor to improve distributed computing performance
*  Add C++ overwrite storage service
*  Add eggroll cleanup API

> Deploy 
* Add auto-deploy 
* Improved deployment documentation

> Example
* Add Hetero Federated Feature Binning example
* Add Hetero Federated Feature Selection example
* Add Hetero DNN example
* Add toy example
* Add task manager examples
* Add Serving example

### Bug Fixes and Other Changes
* Hetero-LR Minibath bugfixed
* Gradient Average bugfixed
* One-second latency for proxy bugfixed
* Training flowid bugfixed
* Many bugfixes
* Many performance improvements
* Many documentation fixes

## Release 0.1

Initial release of FATE.

### Major Features
> WorkFlow
*	Support Intersection workflow
*	Support Train workflow
*	Support Predict workflow
*	Support Validation workflow
*	Support Model Load and Save workflow

> FederatedML
*	Support Distributed Secure Intersection and Raw Intersection for Sample Alignment
*   Support Distributed Homogeneous LR and Heterogeneous LR
*   Support Distributed SecureBoost
*   Support Distributed Secure Federated Transfer Learning
*   Support Binary and Multi-Class Evaluation
*   Support Model Cross-Validation
*	Supprt Mini-Batch
*   Support L1, L2 Regularizers
*   Support Multi-Party Homogeneous FederatedAggregator
*	Support Multi-Party Heterogeneous FederatedAggregator
*	Support Partially Homomorphic Encryption MPC Protocol
	

> Architecture
* Initial release of Computing APIs
* Initial release of Storage APIs
* Initial release of Federation APIs
* Initial release of cross-site network communication (i.e. 'Federation')
* Initial release of Standalone runtime, including computing engine and k-v storage
* Initial release of Distributed runtime, including distributed computing engine, distributed k-v storage, metadata management and intra-site/cross-site network communication
* Support cross-site heterogenous infrastructure
* Initial support of modeling and inference

  
> Deploy
*	Support standalone (docker & manual) deployment
*   Support cluster deployment

