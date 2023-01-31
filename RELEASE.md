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

> Deploy
* Support installing from PyPI
