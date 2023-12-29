# Release 1.0.0
## Major Features and Improvements


* Implement the transmission interface in accordance with the “ Technical Specification for Financial Industry Privacy Computing Interconnection Platform”,The transmission interface is compatible with FATE1. X version and  FATE2. X version

* Supports GRPC synchronous and streaming transmission, supports TLS secure transmission protocol, and is compatible with FATE1. X rollsite components

* Supports Http1. X protocol transmission and TLS secure transmission protocol

* Support message queue mode transmission, used to replace rabbitmq and pulsar components in FATE1. X

* Supports Eggroll and Spark computing engines

* Supports networking as an Exchange component, with support for FATE1. X and FATE2. X access

* Compared to the rollsite component, it improves the exception handling logic during transmission and provides more accurate log output for quickly locating exceptions.

* The routing configuration is basically consistent with the original rollsite, reducing the difficulty of porting

* Supports HTTP interface modification of routing tables and provides simple permission verification

* Improved network connection management logic, reduced connection leakage risk, and improved transmission efficiency

* Using different ports to handle access requests both inside and outside the cluster, facilitating the adoption of different security policies for different ports