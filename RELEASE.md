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

