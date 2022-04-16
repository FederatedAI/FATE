# FATE Test

A collection of useful tools to running FATE's test.

## Testsuite

Testsuite is used for running a collection of jobs in sequence. Data
used for jobs could be uploaded before jobs are submitted and,
optionally, be cleaned after jobs finish. This tool is useful for FATE's
release test.

### command options

```bash
fate_test suite --help
```

1.  include:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json>
    ```
    
    will run testsuites in
    *path1*

2.  exclude:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> -e <path2 to exclude> -e <path3 to exclude> ...
    ```
    
    will run testsuites in *path1* but not in *path2* and *path3*

3.  glob:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> -g "hetero*"
    ```
    
    will run testsuites in sub directory start with *hetero* of
    *path1*

4.  replace:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> -r '{"maxIter": 5}'
    ```
    
    will find all key-value pair with key "maxIter" in
    <span class="title-ref">data conf</span> or
    <span class="title-ref">conf</span> or
    <span class="title-ref">dsl</span> and replace the value with 5

5.  timeout:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> -m 3600
    ```
    
    will run testsuites in *path1* and timeout when job does not finish
    within 3600s; if tasks need more time, use a larger threshold

6.  task-cores
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> -p 4
    ```
    
    will run testsuites in *path1* with EGGROLL "task-cores" set to 4;
    only effective for DSL conf

7.  update-job-parameters
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> -uj {}
    ```
    
    will run testsuites in *path1* with respective job parameters set to
    provided values

8.  update-component-parameters
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> -uc {}
    ```
    
    will run testsuites in *path1* with respective component parameters
    set to provided values

9.  skip-dsl-jobs:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> --skip-dsl-jobs
    ```
    
    will run testsuites in *path1* but skip all *tasks* in testsuites.
    It's would be useful when only pipeline tasks
    needed.

10. skip-pipeline-jobs:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> --skip-pipeline-jobs
    ```
    
    will run testsuites in *path1* but skip all *pipeline tasks* in
    testsuites. It's would be useful when only dsl tasks needed.

11. skip-data:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> --skip-data
    ```
    
    will run testsuites in *path1* without uploading data specified in
    *testsuite.json*.

12. data only:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> --data-only
    ```
    
    will only upload data specified in *testsuite.json* without running
    jobs

13. disable-clean-data:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> --disable-clean-data
    ```
    
    will run testsuites in *path1* without removing data from storage
    after tasks
    finish

14. enable-clean-data:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> --enable-clean-data
    ```
    
    will remove data from storage after finishing running testsuites

15. yes:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json> --yes
    ```
    
    will run testsuites in *path1* directly, skipping double check

### testsuite configuration

Configuration of jobs should be specified in a testsuite whose file name
ends with "\*testsuite.json". For testsuite examples, please refer [dsl
examples](../../examples/dsl/v2) and [pipeline
examples](../../examples/dsl/pipeline).

A testsuite includes the following elements:

  - data: list of local data to be uploaded before running FATE jobs
    
      - file: path to original data file to be uploaded, should be
        relative to testsuite or FATE installation path
      - head: whether file includes header
      - partition: number of partition for data storage
      - table\_name: table name in storage
      - namespace: table namespace in storage
      - role: which role to upload the data, as specified in
        fate\_test.config; naming format is:
        "{role\_type}\_{role\_index}", index starts at 0
    
    ```json
    "data": [
        {
            "file": "examples/data/motor_hetero_host.csv",
            "head": 1,
            "partition": 8,
            "table_name": "motor_hetero_host",
            "namespace": "experiment",
            "role": "host_0"
        }
    ]
    ```

  - tasks: includes arbitrary number of jobs with paths to corresponding
    dsl and conf files
    
      - job: name of job to be run, must be unique within each group
        list
        
          - conf: path to conf filw, should be relative to testsuite
          - dsl: path to dsl file, should be relative to testsuite
        
        
        ```json
        "tasks": {
             "cv": {
                 "conf": "hetero_lr_cv_conf.json",
                 "dsl": "hetero_lr_cv_dsl.json"
             },
             "early-stop": {
                 "conf": "hetero_lr_early_stop_conf.json",
                 "dsl": "hetero_lr_early_stop_dsl.json"
             }
        }
        ```

  - pipeline\_tasks: includes arbitrary number of pipeline jobs with
    paths to corresponding python script
    
      - job: name of job to be run, must be unique within each group
        list
        
          - script: path to pipeline script, should be relative to
            testsuite
        
        
        ```json
        "pipeline_tasks": {
             "cv": {
                 "script": "pipeline-hetero-lr-cv.py"
             },
             "normal": {
                 "script": "pipeline-hetero-lr-early-stop.py"
             }
        }
        ```
    
      - model\_deps(deps): model to be used for prediction task
        
        ```json
        "tasks": {
             "cv": {
                 "conf": "hetero_lr_cv_conf.json",
                 "dsl": "hetero_lr_cv_dsl.json"
             },
             "normal": {
                 "conf": "hetero_lr_normal_conf.json",
                 "dsl": "hetero_lr_normal_dsl.json"
             },
             "predict": {
             "conf": "hetero-lr-normal-predict-conf.json",
             "dsl": "hetero-lr-normal-predict-dsl.json",
             "deps": "normal"
             }
        }
        ```
    
      - data\_deps: component output data from previous task to be used
        as designated input for current task(only used for dsl tasks)
        
        ```json
        "tasks": {
        "column-expand": {
            "conf": "./test_column_expand_job_conf.json",
            "dsl": "./test_column_expand_job_dsl.json"
        },
        "column-expand-train": {
            "conf": "./test_column_expand_train_job_conf.json",
            "dsl": "./test_column_expand_train_job_dsl.json",
            "data_deps": {
                "column-expand": {
                    "guest_0": {
                        "reader_0": "column_expand_0"
                    }
                }
            }
        }
        ```
        
        }
    
      - cache\_deps: component output cache from previous task to be
        used as designated cache loader input for current task(only used
        for intersect dsl tasks)
        
        ```json
        "tasks": {
        "intersect-cache": {
            "conf": "./test_intersect_cache_job_conf.json",
            "dsl": "./test_intersect_cache_job_dsl.json"
        },
        "intersect-cache-loader": {
            "conf": "./test_intersect_cache_loader_job_conf.json",
            "dsl": "./test_intersect_cache_loader_job_dsl.json",
            "cache_deps": {
                "intersect-cache"
            }
        }
        ```
        
        }
    
      - model\_loader\_deps: component output model from previous task
        to be used as designated model loader input for current
        task(only used for dsl tasks)
        
        ```json
        "tasks": {
        "hetero-lr": {
            "conf": "./test_hetero_lr_job_conf.json",
            "dsl": "./test_hetero_lr_job_dsl.json"
        },
        "hetero-lr-model-loader": {
            "conf": "./test_hetero_lr_model_loader_job_conf.json",
            "dsl": "./test_hetero_lr_model_loader_job_dsl.json",
            "model_loader_deps": {
                "hetero-lr"
            }
        }
        ```
        
        }

## Benchmark Quality

Benchmark-quality is used for comparing modeling quality between FATE
and other machine learning systems. Benchmark produces a metrics
comparison summary for each benchmark job group.

Benchmark can also compare metrics of different models from the same
script/PipeLine job. Please refer to the [script writing
guide](#testing-script) below for
instructions.

```bash
fate_test benchmark-quality -i examples/benchmark_quality/hetero_linear_regression
```

```bash
|----------------------------------------------------------------------|
|                             Data Summary                             |
|-------+--------------------------------------------------------------|
|  Data |                         Information                          |
|-------+--------------------------------------------------------------|
| train | {'guest': 'motor_hetero_guest', 'host': 'motor_hetero_host'} |
|  test | {'guest': 'motor_hetero_guest', 'host': 'motor_hetero_host'} |
|-------+--------------------------------------------------------------|


|-------------------------------------------------------------------------------------------------------------------------------------|
|                                                           Metrics Summary                                                           |
|-------------------------------------------+-------------------------+--------------------+---------------------+--------------------|
|                 Model Name                | root_mean_squared_error |      r2_score      |  mean_squared_error | explained_variance |
|-------------------------------------------+-------------------------+--------------------+---------------------+--------------------|
| local-hetero_linear_regression-regression |    0.312552080517407    | 0.9040310440206087 | 0.09768880303575968 | 0.9040312584426697 |
|  FATE-hetero_linear_regression-regression |    0.3139977881119483   | 0.9031411831961411 | 0.09859461093919598 | 0.903146386539082  |
|-------------------------------------------+-------------------------+--------------------+---------------------+--------------------|
|-------------------------------------|
|            Match Results            |
|-------------------------+-----------|
|          Metric         | All Match |
| root_mean_squared_error |    True   |
|         r2_score        |    True   |
|    mean_squared_error   |    True   |
|    explained_variance   |    True   |
|-------------------------+-----------|


|-------------------------------------------------------------------------------------|
|                             FATE Script Metrics Summary                             |
|--------------------+---------------------+--------------------+---------------------|
| Script Model Name  |         min         |        max         |         mean        |
|--------------------+---------------------+--------------------+---------------------|
|  linr_train-FATE   | -1.5305666678748353 | 1.4968292506353484 | 0.03948016870496807 |
| linr_validate-FATE | -1.5305666678748353 | 1.4968292506353484 | 0.03948016870496807 |
|--------------------+---------------------+--------------------+---------------------|
|---------------------------------------|
|   FATE Script Metrics Match Results   |
|----------------+----------------------|
|     Metric     |      All Match       |
|----------------+----------------------|
|      min       |         True         |
|      max       |         True         |
|      mean      |         True         |
|----------------+----------------------|
```

### command options

use the following command to show help message

```bash
fate_test benchmark-quality --help
```

1.  include:
    
    ```bash
    fate_test benchmark-quality -i <path1 contains *benchmark.json>
    ```
    
    will run benchmark testsuites in
    *path1*

2.  exclude:
    
    ```bash
    fate_test benchmark-quality -i <path1 contains *benchmark.json> -e <path2 to exclude> -e <path3 to exclude> ...
    ```
    
    will run benchmark testsuites in *path1* but not in *path2* and
    *path3*

3.  glob:
    
    ```bash
    fate_test benchmark-quality -i <path1 contains *benchmark.json> -g "hetero*"
    ```
    
    will run benchmark testsuites in sub directory start with *hetero*
    of
    *path1*

4.  tol:
    
    ```bash
    fate_test benchmark-quality -i <path1 contains *benchmark.json> -t 1e-3
    ```
    
    will run benchmark testsuites in *path1* with absolute tolerance of
    difference between metrics set to 0.001. If absolute difference
    between metrics is smaller than *tol*, then metrics are considered
    almost equal. Check benchmark testsuite [writing
    guide](#benchmark-testsuite) on setting alternative tolerance.

5.  storage-tag
    
    
    
    ```bash
    fate_test performance -i <path1 contains *benchmark.json> -s test
    ```
    
    will run benchmark testsuites in *path1* with performance stored
    under provided tag for future comparison; note that FATE-Test
    always records the most recent run for each tag; if the same tag
    is used more than once, only metrics from the latest job is
    kept

6.  history-tag
    
    
    
    ```bash
    fate_test performance -i <path1 contains *benchmark.json> -v test1 -v test2
    ```
    
    will run benchmark testsuites in *path1* with performance compared
    to history jobs under provided
    tag(s)

7.  skip-data:
    
    ```bash
    fate_test benchmark-quality -i <path1 contains *benchmark.json> --skip-data
    ```
    
    will run benchmark testsuites in *path1* without uploading data
    specified in
    *benchmark.json*.

8.  disable-clean-data:
    
    ```bash
    fate_test suite -i <path1 contains *benchmark.json> --disable-clean-data
    ```
    
    will run benchmark testsuites in *path1* without removing data from
    storage after tasks
    finish

9.  enable-clean-data:
    
    ```bash
    fate_test suite -i <path1 contains *benchmark.json> --enable-clean-data
    ```
    
    will remove data from storage after finishing running benchmark
    testsuites

10. yes:
    ```bash
    fate_test benchmark-quality -i <path1 contains *benchmark.json> --yes
    ```
 
    will run benchmark testsuites in *path1* directly, skipping double
    check

### benchmark job configuration

Configuration of jobs should be specified in a benchmark testsuite whose
file name ends with "\*benchmark.json". For benchmark testsuite example,
please refer [here](../../examples/benchmark_quality).

A benchmark testsuite includes the following elements:

  - data: list of local data to be uploaded before running FATE jobs
    
      - file: path to original data file to be uploaded, should be
        relative to testsuite or FATE installation path
      - head: whether file includes header
      - partition: number of partition for data storage
      - table\_name: table name in storage
      - namespace: table namespace in storage
      - role: which role to upload the data, as specified in
        fate\_test.config; naming format is:
        "{role\_type}\_{role\_index}", index starts at 0
    
    
    ```json
    "data": [
        {
            "file": "examples/data/motor_hetero_host.csv",
            "head": 1,
            "partition": 8,
            "table_name": "motor_hetero_host",
            "namespace": "experiment",
            "role": "host_0"
        }
    ]
    ```

  - job group: each group includes arbitrary number of jobs with paths
    to corresponding script and configuration
    
      - job: name of job to be run, must be unique within each group
        list
        
          - script: path to [testing script](#testing-script), should be
            relative to testsuite
          - conf: path to job configuration file for script, should be
            relative to testsuite
        
        
        ```json
        "local": {
             "script": "./local-linr.py",
             "conf": "./linr_config.yaml"
        }
        ```
    
      - compare\_setting: additional setting for quality metrics
        comparison, currently only takes `relative_tol`
        
        If metrics *a* and *b* satisfy *abs(a-b) \<= max(relative\_tol
        \* max(abs(a), abs(b)), absolute\_tol)* (from [math
        module](https://docs.python.org/3/library/math.html#math.isclose)),
        they are considered almost equal. In the below example, metrics
        from "local" and "FATE" jobs are considered almost equal if
        their relative difference is smaller than *0.05 \*
        max(abs(local\_metric), abs(pipeline\_metric)*.
    
    
    ```json
    "linear_regression-regression": {
        "local": {
            "script": "./local-linr.py",
            "conf": "./linr_config.yaml"
        },
        "FATE": {
            "script": "./fate-linr.py",
            "conf": "./linr_config.yaml"
        },
        "compare_setting": {
            "relative_tol": 0.01
        }
    }
    ```

### testing script

All job scripts need to have `Main` function as an entry point for
executing jobs; scripts should return two dictionaries: first with data
information key-value pairs: {data\_type}: {data\_name\_dictionary}; the
second contains {metric\_name}: {metric\_value} key-value pairs for
metric comparison.

By default, the final data summary shows the output from the job named
"FATE"; if no such job exists, data information returned by the first
job is shown. For clear presentation, we suggest that user follow this
general [guideline](../../examples/data/README.md#data-set-naming-rule)
for data set naming. In the case of multi-host task, consider numbering
host as such:

    {'guest': 'default_credit_homo_guest',
     'host_1': 'default_credit_homo_host_1',
     'host_2': 'default_credit_homo_host_2'}

Returned quality metrics of the same key are to be compared. Note that
only **real-value** metrics can be compared.

To compare metrics of different models from the same script, metrics of
each model need to be wrapped into dictionary in the same format as the
general metric output above.

In the returned dictionary of script, use reserved key `script_metrics`
to indicate the collection of metrics to be compared.

  - FATE script: `Main` should have three inputs:
      - config: job configuration,
        [JobConfig](../../python/fate_client/pipeline/utils/tools.py#L64)
        object loaded from "fate\_test\_config.yaml"
      - param: job parameter setting, dictionary loaded from "conf" file
        specified in benchmark testsuite
      - namespace: namespace suffix, user-given *namespace* or generated
        timestamp string when using *namespace-mangling*
  - non-FATE script: `Main` should have one or two inputs:
      - param: job parameter setting, dictionary loaded from "conf" file
        specified in benchmark testsuite
      - (optional) config: job configuration,
        [JobConfig](../../python/fate_client/pipeline/utils/tools.py#L64)
        object loaded from "fate\_test\_config.yaml"

Note that `Main` in FATE & non-FATE scripts can also be set to take zero
input argument.

## Benchmark Performance

`Performance` sub-command is used to test
efficiency of designated FATE jobs.

Example tests may be found [here](../../examples/benchmark_performance).

### command options

```bash
fate_test performance --help
```

1.  job-type:
    
    ```bash
    fate_test performance -t intersect
    ```
    
    will run testsuites from intersect sub-directory (set in config) in
    the default performance directory; note that only one of `task` and
    `include` is
    needed

2.  include:
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json>; note that only one of ``task`` and ``include`` needs to be specified.
    ```
    
    will run testsuites in *path1*. Note that only one of `task` and
    `include` needs to be specified; when both are given, path from
    `include` takes
    priority.

3.  replace:
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json> -r '{"maxIter": 5}'
    ```
    
    will find all key-value pair with key "maxIter" in
    <span class="title-ref">data conf</span> or
    <span class="title-ref">conf</span> or
    <span class="title-ref">dsl</span> and replace the value with 5

4.  timeout:
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json> -m 3600
    ```
    
    will run testsuites in *path1* and timeout when job does not finish
    within 3600s; if tasks need more time, use a larger threshold

5.  max-iter:
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json> -e 5
    ```
    
    will run testsuites in *path1* with all values to key "max\_iter"
    set to 5

6.  max-depth
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json> -d 4
    ```
    
    will run testsuites in *path1* with all values to key "max\_depth"
    set to 4

7.  num-trees
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json> -nt 5
    ```
    
    will run testsuites in *path1* with all values to key "num\_trees"
    set to 5

8.  task-cores
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json> -p 4
    ```
    
    will run testsuites in *path1* with EGGROLL "task\_cores" set to 4

9.  update-job-parameters
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json> -uj {}
    ```
    
    will run testsuites in *path1* with respective job parameters set to
    provided values

10. update-component-parameters
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json> -uc {}
    ```
    
    will run testsuites in *path1* with respective component parameters
    set to provided values

11. storage-tag
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json> -s test
    ```
    
    will run testsuites in *path1* with performance time stored under
    provided tag for future comparison; note that FATE-Test always
    records the most recent run for each tag; if the same tag is used
    more than once, only performance from the latest job is
    kept

12. history-tag
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json> -v test1 -v test2
    ```
    
    will run testsuites in *path1* with performance time compared to
    history jobs under provided
    tag(s)

13. skip-data:
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json> --skip-data
    ```
    
    will run testsuites in *path1* without uploading data specified in
    *testsuite.json*.

14. disable-clean-data:
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json> --disable-clean-data
    ```
    
    will run testsuites in *path1* without removing data from storage
    after tasks finish

15. yes:
    
    ```bash
    fate_test performance -i <path1 contains *testsuite.json> --yes
    ```
    
    will run testsuites in *path1* directly, skipping double check

## data

`Data` sub-command is used for upload,
delete, and generate
    dataset.

### data command options

```bash
fate_test data --help
```

1. include:
    
    ```bash
    fate_test data [upload|delete] -i <path1 contains *testsuite.json | *benchmark.json>
    ```
    
    will upload/delete dataset in testsuites in
    *path1*

2. exclude:
    
    ```bash
    fate_test data [upload|delete] -i <path1 contains *testsuite.json | *benchmark.json> -e <path2 to exclude> -e <path3 to exclude> ...
    ```
    
    will upload/delete dataset in testsuites in *path1* but not in
    *path2* and
    *path3*

3. glob:
    
    ```bash
    fate_test data [upload|delete] -i <path1 contains \*testsuite.json | \*benchmark.json> -g "hetero*"
    ```
    
    will upload/delete dataset in testsuites in sub directory start with
    *hetero* of
    *path1*

4. upload example data:

    ```bash
    fate_test data upload -t [min_test|all_examples]
    ```
   
    will upload dataset for min_test or all examples of fate. Once command is executed successfully, 
    you are expected to see the following feedback which showing the table information for you:  

    ```bash
    [2020-06-12 14:19:39]uploading @examples/data/breast_hetero_guest.csv >> experiment.breast_hetero_guest
    [2020-06-12 14:19:39]upload done @examples/data/breast_hetero_guest.csv >> experiment.breast_hetero_guest, job_id=2020061214193960279930
    [2020-06-12 14:19:42]2020061214193960279930 success, elapse: 0:00:02
    [2020-06-12 14:19:42] check_data_out {'data': {'count': 569, 'namespace': 'experiment', 'partition': 16, 'table_name': 'breast_hetero_guest'}, 'retcode': 0, 'retmsg': 'success'}
    ```
   
    Note: uploading configurations are [min_test_config](../../examples/data/upload_config/min_test_data_testsuite.json) and [all_examples](../../examples/data/upload_config/all_examples_data_testsuite.json),
          user can add more data by modifying them or check out the example data's name and namespace.

6. download mnist data:

    ```bash
    fate_test data download -t mnist -o ${mnist_data_dir}
    ```
   
    -t: if not specified, default is "mnist"
    -o: directory of download data, default is "examples/data"

### generate command options

```bash
fate_test data --help
```

1.  include:
    
    ```bash
    fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json>
    ```
    
    will generate dataset in testsuites in *path1*; note that only one
    of `type` and `include` is
    needed

2.  host-data-type:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json | *benchmark.json> -ht {tag-value | dense | tag }
    ```
    
    will generate dataset in testsuites *path1* where host data are of
    selected
    format

3.  sparsity:
    
    ```bash
    fate_test suite -i <path1 contains *testsuite.json | *benchmark.json> -s 0.2
    ```
    
    will generate dataset in testsuites in *path1* with sparsity at 0.1;
    useful for tag-formatted
    data

4.  encryption-type:
    
    ```bash
    fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json> -p {sha256 | md5}
    ```
    
    will generate dataset in testsuites in *path1* with hash id using
    SHA256
    method

5.  match-rate:
    
    ```bash
    fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json> -m 1.0
    ```
    
    will generate dataset in testsuites in *path1* where generated host
    and guest data have intersection rate of
    1.0

6.  guest-data-size:
    
    ```bash
    fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json> -ng 10000
    ```
    
    will generate dataset in testsuites *path1* where guest data each
    have 10000
    entries

7.  host-data-size:
    
    ```bash
    fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json> -nh 10000
    ```
    
    will generate dataset in testsuites *path1* where host data have
    10000
    entries

8.  guest-feature-num:
    
    ```bash
    fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json> -fg 20
    ```
    
    will generate dataset in testsuites *path1* where guest data have 20
    features

9.  host-feature-num:
    
    ```bash
    fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json> -fh 200
    ```
    
    will generate dataset in testsuites *path1* where host data have 200
    features

10. output-path:
    
    ```bash
    fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json> -o <path2>
    ```
    
    will generate dataset in testsuites *path1* and write file to
    *path2*

11. force:
    
    ```bash
    fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json> -o <path2> --force
    ```
    
    will generate dataset in testsuites *path1* and write file to
    *path2*; will overwrite existing file(s) if designated file name
    found under
    *path2*

12. split-host:
    
    ```bash
    fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json> -nh 10000 --split-host
    ```
    
    will generate dataset in testsuites *path1*; 10000 entries will be
    divided equally among all host data
    sets

13. upload-data
    
    ```bash
    fate_test data generate  -i <path1 contains *testsuite.json | *benchmark.json> --upload-data
    ```
    
    will generate dataset in testsuites *path1* and upload generated
    data for all parties to
    FATE

14. remove-data
    
    ```bash
    fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json> --upload-data --remove-data
    ```
    
    (effective with `upload-data` set to True) will delete generated
    data after generate and upload dataset in testsuites
    *path1*

15. use-local-data
    
    ```bash
    fate_test data generate -i <path1 contains *testsuite.json | *benchmark.json> --upload-data  --use-local-data
    ```
    
    (effective with `upload-data` set to True) will generate dataset in
    testsuites *path1* and upload data from local server; use this
    option if flow and data storage are deployed to the same server


## MPC Operation Test
`op-test` sub-command is used to test
efficiency and accuracy of secure multiparty computation protocols like Paillier and SPDZ.

1. paillier:
    
    ```bash
    fate_test op-test paillier
    ```
    
    will generate performance and accuracy table for Paillier Protocol, including encryption，decryption, addition and 
    addition and scalar multiplication.

2. spdz
   
    ```bash
    fate_test op-test spdz
    ```
    
    will generate performance and accuracy table for SPDZ Protocol, including encryption，decryption, addition and 
    addition and scalar multiplication.


## Convert tools

`convert` sub-command is used to convert pipeline to dsl.

1. pipeline-to-dsl

    ```bash
    fate_test convert pipeline-to-dsl -i ${your pipeline file}
    ```
   
    will generate dsl & conf under the same folder of your pipeline file,
    before executing the command, make sure the running data of pipeline is already uploaded.

2. pipeline-testsuite-to-dsl-testsuite

    ```bash
    fate_test convert pipeline-testsuite-to-dsl-testsuite -i {your pipeline testsuite folder}
    ```
    
    will transform pipeline testsuite to dsl testsuite under the {your pipeline testsuite folder}/dsl_testsuite
