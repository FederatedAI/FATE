Example Usage Guide.
====================

We provide here examples of FATE jobs, including FATE-Pipeline scripts,
DSL conf files, and modeling quality comparison tasks

We suggest that user use example-runner tool `FATE-Test <../python/fate_test/README.rst>`__.

Below lists included types of examples.

FATE-Pipeline
-------------

To enhance usability of FATE, starting at FATE-v1.5, FATE provides python APIs.
User may develop federated learning models conveniently with
`FATE-Pipeline <../python/fate_client/pipeline/README.rst>`__.
We provide a host of Pipeline examples for each FATE module and a quick start guide for Pipeline
`here <./pipeline>`__

DSL
---

DSL is the first method FATE provides for constructing federated
modeling jobs. For more information, please refer this guide on
`DSL <../doc/dsl_conf_v2_setting_guide.rst>`__.

Upgraded DSL(DSL v2) by FATE-v1.5 comes with the following major features:

1. Predictions DSL may now be configured through FATE-Flow cli. Please note
   that new DSL no longer supports automatic formation of prediction DSL;
   user needs to first form DSL manually with FATE-Flow cli before running
   prediction task.
2. New components may now be added to prediction DSL;
   for instance, ``evaluation`` module may be added to prediction task.
3. Standardize style of ``role_parameter`` and ``algorithm_parameter``.

For DSL v2 examples, please refer `dsl/v2 <./dsl/v2>`__. For examples of
the older version, please refer `dsl/v1 <./dsl/v1>`__. Please note that
starting at version 1.6, FATE may no longer support DSL v1.


Benchmark Quality
-----------------

Starting at FATE-v1.5, FATE provides modeling quality verifier for comparing modeling
quality of centralized training and FATE federated modeling.
As of v1.5, we have provided quality comparison scripts for the
following common models:

* heterogeneous scenario: LogisticRegression(`benchmark_quality/hetero_lr <./benchmark_quality/hetero_lr>`__),
  SecureBoost(`benchmark_quality/hetero_sbt <./benchmark_quality/hetero_sbt>`__),
  FastSecureBoost(`benchmark_quality/hetero_fast_sbt <./benchmark_quality/hetero_fast_sbt>`__),
  NN(`benchmark_quality/hetero_nn <./benchmark_quality/hetero_nn>`__).
* homogeneous scenario: LogisticRegression(`benchmark_quality/homo_lr <./benchmark_quality/homo_lr>`__),
  SecureBoost(`benchmark_quality/homo_sbt <./benchmark_quality/homo_sbt>`__), NN(`benchmark_quality/homo_nn <./benchmark_quality/homo_nn>`__.

To run the comparison, please refer to the guide on `benchmark_quality <../python/fate_test/README.rst>`__.

Upload Default Data
-------------------

FATE provides a collection of publicly available data at `data <./data>`__ directory,
along with a utility script for uploading all data sets. User may use the provided
script to upload all pre-given data, or modify the corresponding configuration file for uploading
arbitrary data. Please refer `scripts <./scripts/README.rst>`__ for details.


Toy Example
-----------

FATE provides simple toy job for quick experiment when user developing FATE modules
or testing for deployment. For details, please refer `toy_example <./toy_example/README.md>`__.


Min-test
--------

Min-test is used for deployment testing and quick modeling demo. Min-test includes
tasks of hetero Logistic Regression and hetero SecureBoost.
User only needs to configure few simple parameters to run a full modeling job
with FATE. Please refer `min_test_task <./min_test_task/README.rst>`__ for instructions.
