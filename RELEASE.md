# Release 1.0.1
## Bug Fixes
* Fix bugs for evaluation data type 
* Fix bugs for feature binning to take abnormal values into consideration
* Fix bugs for train and eval
* Fix bugs in binning merge
* Fix bugs in Samplers
* Fix federated feature selection feature filter bug
* Support upload file  in version argument
* Support get serviceRoleName from configuration


# Release 1.0
## Major Features and Improvements
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


# Release 0.3.2
## Bug Fixes
* Adjust the Logic of Online Service Module
* Adjust the log format
* Replace the grpc connection pool of the online service module
* Improving Model Processing Details


# Release 0.3.1
## Bug Fixes
* fix feature scale bugs in v0.3
* fix federated feature selection bugs in v0.3

# Release 0.3

## Major Features and Improvements

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

## Bug Fixes and Other Changes
* fix detect onehot max column overflow bug.
* fix dataio dense format not reading host data header bug.
* fix bugs of call of statistics function
* fix bug for federated feature selection that at least one feature remains for each party
* Not allowing so small batch size in LR module for safety consideration.
* fix naming error in federated feature selection module.
* Fix the bug of automated publishing model information in some extreme cases
* Fixed some overflow bugs in fixed-point data
* fix many other bugs.


# Release 0.2

## Major Features and Improvements
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

## Bug Fixes and Other Changes
* Hetero-LR Minibath bugfixed
* Gradient Average bugfixed
* One-second latency for proxy bugfixed
* Training flowid bugfixed
* Many bugfixes
* Many performance improvements
* Many documentation fixes

# Release 0.1

Initial release of FATE.

## Major Features
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

