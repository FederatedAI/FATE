# FATE Test Tutorial

A collection of useful tools to running FATE tests and [:file_folder:examples](../../examples).

## quick start

1. install

    ```bash
    pip install -e python/fate_test
    ```
2.  edit default fate\_test\_config.yaml
    
    ```bash
    # edit priority config file with system default editor
    # filling some field according to comments
    fate_test config edit
    ```

3.  configure FATE-Pipeline and FATE-Flow Commandline server setting


    ```bash
    # configure FATE-Pipeline server setting
    pipeline init --port 9380 --ip 127.0.0.1
    # configure FATE-Flow Commandline server setting
    flow init --port 9380 --ip 127.0.0.1
    ```

4.  run some fate\_test suite
    
    ```bash
    fate_test suite -i <path contains *testsuite.json>
    ```

5.  run some fate\_test benchmark
    
    ```bash
    fate_test benchmark-quality -i <path contains *benchmark.json>
    ```

6.  useful logs or exception will be saved to logs dir with namespace
    shown in last step

## command types

  - [suite](../api/fate_test.md#testsuite): used for running [testsuites](../api/fate_test.md#testsuite-configuration), collection of FATE jobs
    
    ```bash
    fate_test suite -i <path contains *testsuite.json>
    ```
   
  - [data](../api/fate_test.md#data): used for upload, delete, and generate dataset
  
    - [upload/delete data](../api/fate_test.md#data-command-options) command:

      ```bash
      fate_test data [upload|delete] -i <path1 contains *testsuite.json | *benchmark.json>
      ```
    - [upload example data of min_test/all_examples](../api/fate_test.md#data-command-options) command:

      ```bash
      fate_test data upload -t min_test
      fate_test data upload -t all_examples
      ```

    - [generate data](../api/fate_test.md#generate-command-options) command:
    
      ```bash
      fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json>
      ```
    
  - [benchmark-quality](../api/fate_test.md#benchmark-quality): used for comparing modeling quality between FATE
    and other machine learning systems, as specified in [benchmark job configuration](../api/fate_test.md#benchmark-job-configuration)
    
    ```bash
    fate_test bq -i <path contains *benchmark.json>
    ```
    
  - [benchmark-performance](../api/fate_test.md#benchmark-performance): used for checking FATE algorithm performance; user
    should first generate and upload data before running performance testsuite

    ```bash
    fate_test data generate -i <path contains *benchmark.json> -ng 10000 -fg 10 -fh 10 -m 1.0 --upload-data
    fate_test performance -i <path contains *benchmark.json> --skip-data
    ```
    
  - [op-test](../api/fate_test.md#mpc-operation-test): used for testing FATE's mpc protocol. 

    ```bash
    fate_test op-test paillier
    fate_test op-test spdz
    ```

  - [convert](../api/fate_test.md#convert-tools): used for converting pipeline to dsl&conf. 

    ```bash
    fate_test convert pipeline-to-dsl -i ${your pipeline file}
    fate_test convert pipeline-testsuite-to-dsl-testsuite -i {your pipeline testsuite folder}
    ```
  

## Usage 

![tutorial](../images/tutorial.gif)
