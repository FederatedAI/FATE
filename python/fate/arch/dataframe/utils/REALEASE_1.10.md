## Release 1.10.0
### Major Features and Improvements
> FederatedML
* Renewed Homo NN: PyTorch-based, support flexible model building:
   * Support user access to complex self-defined PyTorch models or ready-to-use PyTorch models such as DeepFM, ResNet, BERT, Yolo
   * Support various data set types, may build data set based on PyTorch Dataset
   * User-defined training loss
   * User-defined training process: user-defined aggregation algorithm for client and server
   * Provide API for developing Aggregator
* Upgraded Hetero NN: support flexible model building and various data set types:
   * more flexible pytorch top/bottom model customization; provide access to industry approved PyTorch models
   * User-defined training loss
   * Support various data set types, may build data set based on PyTorch Dataset
* FedIPR(experimental): an ownership verification method to protect the intellectual property (IP) of Homogeneous Federated Deep Neural Network.
* Semi-Supervised Algorithm Positive Unlabeled Learning
* Hetero LR & Hetero SecureBoost now supports Intel IPCL
* Intersection support Multi-host Elliptic-curve-based PSI
* Intersection may compute Multi-host Secure PSI Cardinality
* Hetero Feature Optimal Binning now record & show Gini/KS/Chi-Square metrics
* Host may load Hetero Binning model with WOE score through Model Loader
* Hetero Feature Binning support binning by user-provided split points
* Sampler support weighted sampling by instance weight

> FATE-Flow
* Add connection test API
* May configure gRPC message size limit 
* Fix module duplication issue in model 

> FATE-Board
* Display SBT leaf node data
* Support result summary display for Sampler's new method 
* Add model summary for new module Positive Unlabeled Learning
* Improved table display for Binning
* Data filtering on requested model proto
* Adjusted Design
* Improved Logging display adaptation

> Fate-Client
* Flow CLI adds min-test options
* Pipeline adds `data-bind` API, useful for local development 
* Pipeline may reconfigure role/model_id/model_version, switching `party_id` for prediction task
