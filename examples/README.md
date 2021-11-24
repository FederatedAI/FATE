# Example Usage Guide
[中文](README.zh.md)

We provide here examples of FATE jobs, including FATE-Pipeline scripts,
DSL conf files, and modeling quality comparison tasks

For auto-test of FATE, FATE provides auto-test tools [FATE-Test](../doc/api/fate_test.md).

To make it easy to submit FL modeling tasks using Pipeline or DSL, we recommend that users install 
[FATE-Client](../doc/api/fate_client/pipeline.md).

To quickly start to use pipeline or dsl,
please refer to [tutorial](../doc/tutorial/README.md)


Below lists included types of examples.

## FATE-Pipeline

To enhance usability of FATE, starting at FATE-v1.5, FATE provides 
Pipeline Module. User may develop federated learning models conveniently using python API.
Please refer this: [FATE-Pipeline](../doc/api/fate_client/pipeline.md). We provide many [Pipeline
examples](./pipeline) for each FATE module.

## DSL
DSL is language of building federated modeling tasks based on configuration file for FATE.
For more information, please refer this
[dsl setting guide](../doc/tutorial/dsl_conf/dsl_conf_v2_setting_guide.md) on DSL.

Upgraded DSL(DSL v2) by FATE-v1.5 comes with the following major
features:

1.  Predictions DSL may now be configured through FATE-Flow cli. Please
    note that with new DSL training job will no longer automatically
    form prediction DSL; user needs to first form DSL manually with
    FATE-Flow cli before running prediction task.
2.  New components may now be added to prediction DSL; for instance,
    `evaluation` module may be added to prediction task.
3.  Standardize style of `role_parameter` and `algorithm_parameter`.

For DSL v2 examples, please refer [dsl/v2](./dsl/v2). Please note that starting
at version 1.7, FATE may no longer support DSL/v1 and remove related examples. 
However, tools will be provided to transform DSL/ V1 built models into DSL/ V2 to facilitate model migration to DSL/ V2.  


## Benchmark Quality

Starting at FATE-v1.5, FATE provides modeling quality verifier for
comparing modeling quality of centralized training and FATE federated
modeling. We have provided quality comparison scripts for
the following common models:

- heterogeneous scenario:
    - LogisticRegression([benchmark\_quality/hetero\_lr](./benchmark_quality/hetero_lr))
    - LinearRegression([benchmark\_quality/hetero\_linear_regression](./benchmark_quality/hetero_linear_regression))
    - SecureBoost([benchmark\_quality/hetero\_sbt](./benchmark_quality/hetero_sbt))
    - FastSecureBoost([benchmark\_quality/hetero\_fast\_sbt](./benchmark_quality/hetero_fast_sbt))
    - NN([benchmark\_quality/hetero\_nn](./benchmark_quality/hetero_nn))
- homogeneous scenario:
    - LogisticRegression([benchmark\_quality/homo\_lr](./benchmark_quality/homo_lr))
    - SecureBoost([benchmark\_quality/homo\_sbt](./benchmark_quality/homo_sbt))
    - NN([benchmark\_quality/homo\_nn](./benchmark_quality/homo_nn)

Starting at v1.6, benchmark quality supports matching metrics from the
same script. For more details, please refer to the
[benchmark-quality guide](../doc/api/fate_test.md#benchmark-quality).


## Benchmark Performance

FATE-Test also provides performance benchmark for FATE federated
modeling. We include benchmark test suites for the following common models:

  - Hetero Logistic Regression([benchmark\_performance/hetero\_lr](./benchmark_performance/hetero_lr)),
  - Hetero SecureBoost([benchmark\_performance/hetero\_sbt](./benchmark_performance/hetero_sbt)),
  - Hetero SSHE LR([benchmark\_performance/hetero\_fast\_sbt](./benchmark_performance/hetero_sshe_lr)),
  - Hetero Intersect:
    - [benchmark\_performance/intersect_single](./benchmark_performance/intersect_single)
    - [benchmark\_performance/intersect_multi](./benchmark_performance/intersect_multi)
  
For more details, please refer to the
[benchmark-performance guide](../doc/api/fate_test.md#benchmark-performance).


## Upload Default Data
user may use [FATE-Test](../doc/api/fate_test.md#data) for uploading data.

## Toy Example

FATE provides simple toy example for quick testing the network connectivity between sites
Such as 
```
flow test toy -gid 9999 -hid 10000
```

## Min-test

Min-test is used for deployment testing and quick modeling demo.
Min-test includes tasks of hetero Logistic Regression and hetero
SecureBoost. User only needs to configure few simple parameters to run a
full modeling job with FATE. Please refer
[min\_test\_task](./min_test_task/README.rst) for instructions.
