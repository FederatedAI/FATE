## Release
### Major Features and Improvements
> Arch 2.0：Building Unified and Standardized API for Heterogeneous Computing Engines Interconnection
* Introduce `Context` to manage useful APIs for developers, such as `Distributed Compting`, `Federation`, `Cipher`, `Tensor`, `Metrics`,  and `IO`.
* Introduce `Tensor` data structure to handle local and distributed matrix operation, with built-in heterogeneous acceleration support. 
  * abstracted PHETensor, smooth switch between various underlying PHE implementations through standard interface
* Introduce `DataFrame`, a 2D tabular data structure for data io and simple feature engineering
  * add data block manager to support mixed-type columns & feature anonymization
  * added 30+ operator interfaces for statistics, including comparison, indexing, data binning, and transformation, etc
* Refactor `Federation`, a unified interface for federated computing. We provide a unified Serdes control and more user-friendly api.
* Introduce `Config`, a unified configuration for FATE, including safety restrictions, system configuration, and algorithm configuration
* Refactor `logger`, customizable logging for different use cases and flavors.
* Introduce `Launcher`, a simple tool for federated program execution, especially useful for standalone and local debugging
* Framework: PSI-ECDH protocol support, single entry for histogram statistical computation 
* Deepspeed integration: support distributed training using deepspeed with Eggroll.
* Protocol: Support for SSHE(mpc and homomophic encryption mixed protocol), ECDH, Secure Aggregation protocols
* Experimental Integrate `Crypten` for SMPC support, more protocols and features will be added in the future

> Components 2.0: Building Standardized Algorithm Components for different Scheduling Engines
* Introduce components toolbox to wrap ML modules as standard executable programs
* spec and loader expose clear API for smooth internal extension and external system integration
* Provide several cli tools to interact and execute components
* Input-Output: Further decoupling of FATE-Flow, providing standardized black-box calling processes 
* Component Definition: Support for typing-based definition, automatic checking for component parameters, support for multiple types of data and model input and output, in addition to multiple inputs

> ML 2.0: Major functionality migration from FATE-v1.x, decoupling call hierarchy
* Data preprocessing: Added DataFrame Transformer; Reader, Union and DataSplit migration completed
* Feature Engineering: Migrated HeteroFederatedBinning, HeteroFeatureSelection, DataStatistics, Sampling, FeatureScale and Pearson Correlation
* Federated Training Migrated: HeteroSecureBoost, HomoNN, HeteroCoordinatedLogisticRegression, HeteroCoordinatedLinearRegression, SSHE-LogisticRegression and SSHE-LinearRegression
* Federated Training Added: 
  * SSHE-HeteroNN: based on mpc and homomorphic encryption mixed protocal
  * FedPASS-HeteroNN: based on fedpass protocol

> OSX(Open Site Exchange) 1.0: Building Open Platform for Cross-Site Communication Interconnection 
* Implement the transmission interface in accordance with the “ Technical Specification for Financial Industry Privacy Computing Interconnection Platform”,The transmission interface is compatible with FATE 1.X version and  FATE 2.X version
* Supports GRPC synchronous and streaming transmission, supports TLS secure transmission protocol, and is compatible with FATE1.X rollsite components
* Supports Http 1.X protocol transmission and TLS secure transmission protocol
* Support message queue mode transmission, used to replace rabbitmq and pulsar components in FATE 1.X
* Supports Eggroll and Spark computing engines
* Supports networking as an Exchange component, with support for FATE 1.X and FATE 2.X access
* Compared to the rollsite component, it improves the exception handling logic during transmission and provides more accurate log output for quickly locating exceptions.
* The routing configuration is basically consistent with the original rollsite, reducing the difficulty of porting
* Supports HTTP interface modification of routing tables and provides simple permission verification
* Improved network connection management logic, reduced connection leakage risk, and improved transmission efficiency
* Using different ports to handle access requests both inside and outside the cluster, facilitating the adoption of different security policies for different ports

> FATE Flow 2.0: Building Open and Standardized Scheduling Platform for Scheduling Interconnection
* Migrated functions: data upload/download, process scheduling, component output data/model/metric management, multi-storage adaptation for models, authentication, authorization, feature anonymization, multi-computing/storage/communication engine adaptation, and system high availability
* Optimized process scheduling, with scheduling separated and customizable, and added priority scheduling
* Optimized algorithm component scheduling, dividing execution steps into preprocessing, running, and post-processing
* Optimized multi-version algorithm component registration, supporting registration for mode of components
* Optimized client authentication logic, supporting permission management for multiple clients
* Optimized RESTful interface, making parameter fields and types, return fields, and status codes clearer
* Decoupling the system layer from the algorithm layer, with system configuration moved from the FATE repository to the Flow repository
* Published FATE Flow package to PyPI and added service-level CLI for service management


> Fate-Client 2.0: Building Scalable Federated DSL for Application Layer Interconnection And Providing Tools For Fast Federated Modeling.
* Introduce new scalable and standardized federated DSL IR(Intermediate Representation) for federated modeling job
* Compile python client to DSL IR
* Federated DSL IR extension enhancement: supports multi-party asymmetric scheduling
* Support mutual translation between Standardized Fate-2.0.0 DSL IR and BFIA protocol
* Support using components with BFIA protocol through adapter mode
* Migrated Flow CLI and Flow SDK

> Fate-Test: FATE Automated Testing Tool
* Migrated automated testing for functionality, performance, and correctness


