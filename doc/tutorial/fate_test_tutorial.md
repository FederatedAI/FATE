# FATE Test Tutorial

A collection of useful tools to running FATE tests and [examples](../../examples).

## quick start
1.  edit default fate\_test\_config.yaml
    
    ``` sourceCode bash
    # edit priority config file with system default editor
    # filling some field according to comments
    fate_test config edit
    ```

2.  configure FATE-Pipeline and FATE-Flow Commandline server setting

<!-- end list -->

``` sourceCode bash
# configure FATE-Pipeline server setting
pipeline init --port 9380 --ip 127.0.0.1
# configure FATE-Flow Commandline server setting
flow init --port 9380 --ip 127.0.0.1
```

3.  run some fate\_test suite
    
    ``` sourceCode bash
    fate_test suite -i <path contains *testsuite.json>
    ```

4.  run some fate\_test benchmark
    
    ``` sourceCode bash
    fate_test benchmark-quality -i <path contains *benchmark.json>
    ```

5.  useful logs or exception will be saved to logs dir with namespace
    shown in last step

## command types

  - [suite](../api/fate_test.md#testsuite): used for running [testsuites](../api/fate_test.md#testsuite-configuration), collection of FATE jobs
    
    ``` sourceCode bash
    fate_test suite -i <path contains *testsuite.json>
    ```
   
  - [data](../api/fate_test.md#data): used for upload, delete, and generate dataset
  
    - [upload/delete data](../api/fate_test.md#data-command-options) command:

      ``` sourceCode bash
      fate_test data [upload|delete] -i <path1 contains *testsuite.json | *benchmark.json>
      ```
    - [generate data](../api/fate_test.md#generate-command-options) command:
    
      ``` sourceCode bash
      fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json>
      ```
    
  - [benchmark-quality](../api/fate_test.md#benchmark-quality): used for comparing modeling quality between FATE
    and other machine learning systems, as specified in [benchmark job configuration](../api/fate_test.md#benchmark-job-configuration)
    
    ``` sourceCode bash
    fate_test bq -i <path contains *benchmark.json>
    ```
    
  - [benchmark-performance](../api/fate_test.md#benchmark-performance): used for checking FATE algorithm performance; user
    should first generate and upload data before running performance testsuite

    ``` sourceCode bash
    fate_test data generate -i <path contains *benchmark.json> -ng 10000 -fg 10 -fh 10 -m 1.0 --upload-data
    fate_test performance -i <path contains *benchmark.json>
    ```
    
## Usage 

![tutorial](../images/tutorial.gif)
