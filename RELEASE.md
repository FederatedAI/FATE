## Release 2.0.0-alpha
### Feature Highlights
> Arch
* Introduce `Context` to manage useful APIs for developers, such as `Metrics`, `Cipher`, `Tensor` and `IO`.
* Introduce `Tensor` data structure to handle local and distributed matrix operation, with built-in heterogeneous acceleration support. 
* Introduce `DataFrame`, a 2D tabular data structure for data io and simple feature engineering.
* Refactor `logger`, customizable logging for different use cases and flavors.
* Introduce new federation API suite: `context.<role>.get(name)/context.<role>.put(name=value)`.
* Introduce new federation engine `OSX`

> Components
* Introduce `components` toolbox to wrap `ML` modules as standard executable programs.
* Implement ready-to-use components: reader, intersection, feature scale, lr and evaluation. 
* `spec` and `loader` expose clear `API` for smooth internal extension and external system integration. 
* Provide several cli tools to interact and execute components.

> ML
* Provide base demos for component development: reader、intersection、feature scale、lr and evaluation.

> Client
* Newly designed federated modeling job DAG, more standardized and scalable, more cross-platform friendly
* Support modeling federated tasks directly using local data storage after installation
* Support multiple execution backends, including standalone and Fate-Flow

> Deploy
* Support installing from PyPI
