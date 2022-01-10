# Benchmark Performance Examples

We provide here a host of performance benchmark tests. 

Follow the following steps to run benchmark tests:

1. Generate & upload data using [FATE-Test](../../doc/api/fate_test.md#data):

    ``` sourceCode bash
    fate_test data generate -i examples/benchmark_performance/hetero_lr/hetero_lr_testsuite.json -ng 1000 -nh 1000 -fg 20 -fh 200 -o examples/data/ --upload-data    
    ```
2. Run Test Task using [FATE-Test](../../doc/api/fate_test.md#performance)

    ``` sourceCode bash
    fate_test suite -i examples/benchmark_performance/hetero_lr/hetero_lr_testsuite.json -m 360000 --skip-data
    ```

### Example Tests

1. [Hetero Logistic Regression](hetero_lr)

2. [Hetero Secureboost](hetero_sbt)

3. [Hetero SSHE Logistic Regression](hetero_sshe_lr)

4. [Intersection with Multiple Components](intersect_multi)

5. [Intersection with Single Component](intersect_single)

Users can also run jobs directly with following command:

    flow job submit -c ${runtime_config} -d ${dsl}
