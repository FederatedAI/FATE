## Release 2.0.0-beta
### Major Features and Improvements
> Arch 2.0：Building Unified and Standardized API for Heterogeneous Computing Engines Interconnection
* Framework: PSI-ECDH protocol support, single entry for histogram statistical computation 
* Protocol: Support for ECDH, Secure Aggregation protocols
* Tensor: abstracted PHETensor, smooth switch between various underlying PHE implementations through standard interface
* DataFrame: New data block manager supports mixed-type columns & feature anonymization; added 30+ operator interfaces for statistics, including comparison, indexing, data binning, and transformation, etc.
* Enhanced workflow: Support for Cross Validation workflow

> Components 2.0: Building Standardized Algorithm Components for different Scheduling Engines
* Input-Output: Further decoupling of FATE-Flow, providing standardized black-box calling processes 
* Component Definition: Support for typing-based definition, automatic checking for component parameters, support for multiple types of data and model input and output, in addition to multiple inputs

> ML:-2.0:  Major functionality migration from FATE-v1.x, decoupling call hierarchy 
* Data preprocessing: Added DataFrame Transformer, Union and DataSplit migration completed
* Feature Engineering: Migrated HeteroFederatedBinning, HeteroFeatureSelection, DataStatistics, Sampling, FeatureScale
3. Federated Training: Migrated HeteroSecureBoost, HomoNN, vertical CoordinatedLogisticRegression, and CoordinatedLinearRegression
4. Evaluation: Migrated Evaluation

> OSX(Open Site Exchange) 1.0: Building Open Platform for Cross-Site Communication Interconnection 
* Improved HTTP/1.X protocol support, support for GRPC-to-HTTP transmission 
* Support for TLS secure transmission protocol 
* Added routing table configuration interface 
* Added routing table connectivity automatic check 
* Improved transmission function in cluster mode 
* Enhanced flow control in cluster mode 
* Support for simple interface authentication

> FATE Flow 2.0: Building Open and Standardized Scheduling Platform for Scheduling Interconnection
* Migrated functions: data upload/download, process scheduling, component output data/model/metric management, multi-storage adaptation for models, authentication, authorization, feature anonymization, multi-computing/storage/communication engine adaptation, and system high availability
* Optimized process scheduling, with scheduling separated and customizable, and added priority scheduling
* Optimized algorithm component scheduling, dividing execution steps into preprocessing, running, and post-processing
* Optimized multi-version algorithm component registration, supporting registration for mode of components
* Optimized client authentication logic, supporting permission management for multiple clients
* Adapted to federated communication engine OSX
* Optimized RESTful interface, making parameter fields and types, return fields, and status codes clearer
* Decoupling the system layer from the algorithm layer, with system configuration moved from the FATE repository to the Flow repository
* Published FATE Flow package to PyPI and added service-level CLI for service management

> Fate-Test: FATE Automated Testing Tool
* Migrated automated testing for functionality, performance, and correctness


## Release 2.0.0-alpha
### Feature Highlights
> Arch 2.0：Building Unified and Standardized API for Heterogeneous Computing Engines Interconnection
* Introduce `Context` to manage useful APIs for developers, such as `Metrics`, `Cipher`, `Tensor` and `IO`.
* Introduce `Tensor` data structure to handle local and distributed matrix operation, with built-in heterogeneous acceleration support. 
* Introduce `DataFrame`, a 2D tabular data structure for data io and simple feature engineering.
* Refactor `logger`, customizable logging for different use cases and flavors.
* Introduce new high-level federation API suite: `context.<role>.get(name)/context.<role>.put(name=value)`.

> Components 2.0: Building Standardized Algorithm Components for different Scheduling Engines
* Introduce `components` toolbox to wrap `ML` modules as standard executable programs.
* `spec` and `loader` expose clear `API` for smooth internal extension and external system integration. 
* Provide several cli tools to interact and execute components.
* Implement base demos components: reader, intersection, feature scale, lr and evaluation. 

> ML 2.0(demo)
* Provide base demos for federated machine learning algorithm: intersection、feature scale、lr and evaluation.

> Pipeline 2.0: Building Scalable Federated DSL for Application Layer Interconnection
* Introduce new scalable and standardized federated DSL IR(Intermediate Representation) for federated modeling job
* Compile python client to DSL IR
* Support multiple scalable execution backends, including standalone and Fate-Flow.

> OSX(Open Site Exchange) 1.0: Building Open Platform for Cross-Site Communication Interconnection
* Standardized Cross-Site lower-level federation api
* Support grpc synchronous transmission and streaming transmission; Compatible with eggroll interface and can replace FATE-1.x rollsite component
* Support asynchronous message transmission, which can replace rabbitmq and pulsar components in FATE-1.x
* Support HTTP-1.X protocol transmission
* Support cluster deployment and inter-site traffic control
* Support networking as an Exchange component

> FATE Flow 2.0: Building Open and Standardized Scheduling Platform for Scheduling Interconnection
* Adapted to new scalable and standardized federated DSL IR
* Standardized API interface with param type checking 
* Decoupling Flow from FATE repository 
* Optimized scheduling logic, with configurable dispatcher decoupled from initiator
* Support container-level algorithm loading and task scheduling, enhancing support for cross-platform heterogeneous scenarios
* Independent maintenance for system configuration to enhance flexibility and ease of configuration
* Support new communication engine OSX, while compatible with all engines from Flow 1.X
* Introduce OFX(Open Flow Exchange) module: encapsulated scheduling client to allow cross-platform scheduling

> Deploy
* Support installing from PyPI
