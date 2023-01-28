## Release 2.0.0-alpha
### Feature Highlights
> Arch
* Introduce `Context` to manage useful APIs for developers, such as `Metrics`, `Cipher`, `Tensor` and `IO`.
* Introduce `Tensor` data structure to handle local and distributed matrix operation, with built-in heterogeneous acceleration support. 
* Introduce `DataFrame`, a 2D tabular data structure for data io and simple feature engineering.
* Refactor `logger`, customizable logging for different use cases and flavors.
* Introduce new High-Level federation API suite: `context.<role>.get(name)/context.<role>.put(name=value)`.


> Components
* Introduce `components` toolbox to wrap `ML` modules as standard executable programs.
* Implement base demos components: reader, intersection, feature scale, lr and evaluation. 
* `spec` and `loader` expose clear `API` for smooth internal extension and external system integration. 
* Provide several cli tools to interact and execute components.

> ML
* Provide base demos for component development: reader、intersection、feature scale、lr and evaluation.

> FATE-Client
* Newly designed federated modeling job DAG, more standardized and scalable, more cross-platform friendly
* Support modeling federated tasks directly using local data storage after installation
* Support multiple execution backends, including standalone and Fate-Flow

> OSX(Open Site Exchange)
* Support grpc synchronous transmission and streaming transmission. Compatible with eggroll interface and can replace FATE-1.x rollsite component
* Support asynchronous message transmission, which can replace rabbitmq and pulsar components in FATE-1.x
* Support HTTP-1.X protocol transmission
* Support cluster deployment and inter-site traffic control
* Support networking as an Exchange component

> Deploy
* Support installing from PyPI
