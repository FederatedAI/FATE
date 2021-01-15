FATE Test
=========

A collection of useful tools to running FATE's test.

.. image:: images/tutorial.gif
   :align: center
   :alt: tutorial

quick start
-----------

1. (optional) create virtual env

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate
      pip install -U pip


2. install fate_test

   .. code-block:: bash

      pip install fate_test
      fate_test --help


3. edit default fate_test_config.yaml

   .. code-block:: bash

      # edit priority config file with system default editor
      # filling fields following comments
      fate_test config edit

4. configure FATE-Pipeline and FATE-Flow Commandline server setting

.. code-block:: bash

      # configure FATE-Pipeline server setting
      pipeline config --port 9380 --ip 127.0.0.1
      # configure FATE-Flow Commandline server setting
      flow init --port 9380 --ip 127.0.0.1

5. run some fate_test suite

   .. code-block:: bash

      fate_test suite -i <path contains *testsuite.json>


6. run some fate_test benchmark

   .. code-block:: bash

      fate_test benchmark-quality -i <path contains *benchmark.json>

7. useful logs or exception will be saved to logs dir with namespace shown in last step

develop install
---------------
It is more convenient to use the editable mode during development: replace step 2 with flowing steps

.. code-block:: bash

   pip install -e ${FATE}/python/fate_client && pip install -e ${FATE}/python/fate_test



command types
-------------

- suite: used for running testsuites, collection of FATE jobs

  .. code-block:: bash

     fate_test suite -i <path contains *testsuite.json>


- benchmark-quality used for comparing modeling quality between FATE and other machine learning systems

  .. code-block:: bash

      fate_test benchmark-quality -i <path contains *benchmark.json>



configuration by examples
--------------------------

1. no need ssh tunnel:

   - 9999, service: service_a
   - 10000, service: service_b

   and both service_a, service_b can be requested directly:

   .. code-block:: yaml

      work_mode: 1 # 0 for standalone, 1 for cluster
      data_base_dir: <path_to_data>
      parties:
        guest: [10000]
        host: [9999, 10000]
        arbiter: [9999]
      services:
        - flow_services:
          - {address: service_a, parties: [9999]}
          - {address: service_b, parties: [10000]}

2. need ssh tunnel:

   - 9999, service: service_a
   - 10000, service: service_b

   service_a, can be requested directly while service_b don't,
   but you can request service_b in other node, say B:

   .. code-block:: yaml

      work_mode: 0 # 0 for standalone, 1 for cluster
      data_base_dir: <path_to_data>
      parties:
        guest: [10000]
        host: [9999, 10000]
        arbiter: [9999]
      services:
        - flow_services:
          - {address: service_a, parties: [9999]}
        - flow_services:
          - {address: service_b, parties: [10000]}
          ssh_tunnel: # optional
          enable: true
          ssh_address: <ssh_ip_to_B>:<ssh_port_to_B>
          ssh_username: <ssh_username_to B>
          ssh_password: # optional
          ssh_priv_key: "~/.ssh/id_rsa"


Testsuite
---------

Testsuite is used for running a collection of jobs in sequence. Data used for jobs could be uploaded before jobs are
submitted, and are cleaned when jobs finished. This tool is useful for FATE's release test.

command options
~~~~~~~~~~~~~~~

.. code-block:: bash

      fate_test suite --help

1. include:

   .. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json>

   will run testsuites in *path1*

2. exclude:

   .. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json> -e <path2 to exclude> -e <path3 to exclude> ...

   will run testsuites in *path1* but not in *path2* and *path3*

3. glob:

   .. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json> -g "hetero*"

   will run testsuites in sub directory start with *hetero* of *path1*

4. replace:

   .. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json> -r '{"maxIter": 5}'

   will find all key-value pair with key "maxIter" in `data conf` or `conf` or `dsl` and replace the value with 5


5. skip-data:

   .. code-block:: bash

       fate_test suite -i <path1 contains *testsuite.json> --skip-data

   will run testsuites in *path1* without uploading data specified in *benchmark.json*.


6. yes:

   .. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json> --yes

   will run testsuites in *path1* directly, skipping double check

7. skip-dsl-jobs:

   .. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json> --skip-dsl-jobs

   will run testsuites in *path1* but skip all *tasks* in testsuites. It's would be useful when only pipeline tasks needed.

8. skip-pipeline-jobs:

   .. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json> --skip-pipeline-jobs

   will run testsuites in *path1* but skip all *pipeline tasks* in testsuites. It's would be useful when only dsl tasks needed.


Benchmark Quality
------------------

Benchmark-quality is used for comparing modeling quality between FATE
and other machine learning systems. Benchmark produces a metrics comparison
summary for each benchmark job group.

Benchmark can also compare metrics of different models from the same script/PipeLine job.
Please refer to the `script writing guide <#testing-script>`_ below for instructions.

.. code-block:: bash

   fate_test benchmark-quality -i examples/benchmark_quality/hetero_linear_regression

.. code-block:: bash

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
    |-------------------------+-----------|
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

command options
~~~~~~~~~~~~~~~

use the following command to show help message

.. code-block:: bash

      fate_test benchmark-quality --help

1. include:

   .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json>

   will run benchmark testsuites in *path1*

2. exclude:

   .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> -e <path2 to exclude> -e <path3 to exclude> ...

   will run benchmark testsuites in *path1* but not in *path2* and *path3*

3. glob:

   .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> -g "hetero*"

   will run benchmark testsuites in sub directory start with *hetero* of *path1*

4. tol:

   .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> -t 1e-3

   will run benchmark testsuites in *path1* with absolute tolerance of difference between metrics set to 0.001.
   If absolute difference between metrics is smaller than *tol*, then metrics are considered
   almost equal. Check benchmark testsuite `writing guide <#benchmark-testsuite>`_ on setting alternative tolerance.

5. skip-data:

   .. code-block:: bash

       fate_test benchmark-quality -i <path1 contains *benchmark.json> --skip-data

   will run benchmark testsuites in *path1* without uploading data specified in *benchmark.json*.


6. yes:

   .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> --yes

   will run benchmark testsuites in *path1* directly, skipping double check


benchmark testsuite
~~~~~~~~~~~~~~~~~~~

Configuration of jobs should be specified in a benchmark testsuite whose file name ends
with "\*benchmark.json". For benchmark testsuite example,
please refer `here <../../examples/benchmark_quality>`_.

A benchmark testsuite includes the following elements:

- data: list of local data to be uploaded before running FATE jobs

  - file: path to original data file to be uploaded, should be relative to testsuite or FATE installation path
  - head: whether file includes header
  - partition: number of partition for data storage
  - table_name: table name in storage
  - namespace: table namespace in storage
  - role: which role to upload the data, as specified in fate_test.config;
    naming format is: "{role_type}_{role_index}", index starts at 0

  .. code-block:: json

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

- job group: each group includes arbitrary number of jobs with paths to corresponding script and configuration

  - job: name of job to be run, must be unique within each group list

    - script: path to `testing script <#testing-script>`_, should be relative to testsuite
    - conf: path to job configuration file for script, should be relative to testsuite

    .. code-block:: json

       "local": {
            "script": "./local-linr.py",
            "conf": "./linr_config.yaml"
       }

  - compare_setting: additional setting for quality metrics comparison, currently only takes ``relative_tol``

    If metrics *a* and *b* satisfy *abs(a-b) <= max(relative_tol \* max(abs(a), abs(b)), absolute_tol)*
    (from `math module <https://docs.python.org/3/library/math.html#math.isclose>`_),
    they are considered almost equal. In the below example, metrics from "local" and "FATE" jobs are
    considered almost equal if their relative difference is smaller than
    *0.05 \* max(abs(local_metric), abs(pipeline_metric)*.

  .. code-block:: json

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


testing script
~~~~~~~~~~~~~~

All job scripts need to have ``Main`` function as an entry point for executing jobs; scripts should
return two dictionaries: first with data information key-value pairs: {data_type}: {data_name_dictionary};
the second contains {metric_name}: {metric_value} key-value pairs for metric comparison.

By default, the final data summary shows the output from the job named "FATE"; if no such job exists,
data information returned by the first job is shown. For clear presentation, we suggest that user follow
this general `guideline <../../examples/data/README.md#data-set-naming-rule>`_ for data set naming. In the case of multi-host
task, consider numbering host as such:

::

    {'guest': 'default_credit_homo_guest',
     'host_1': 'default_credit_homo_host_1',
     'host_2': 'default_credit_homo_host_2'}

Returned quality metrics of the same key are to be compared.
Note that only **real-value** metrics can be compared.

To compare metrics of different models from the same script,
metrics of each model need to be wrapped into dictionary in the same format as the general metric output above.

In the returned dictionary of script, use reserved key ``script_metrics`` to indicate the collection of groups of metrics to be compared.

- FATE script: ``Main`` always has three inputs:

  - config: job configuration, `JobConfig <../fate_client/pipeline/utils/tools.py#L64>`_ object loaded from "fate_test_config.yaml"
  - param: job parameter setting, dictionary loaded from "conf" file specified in benchmark testsuite
  - namespace: namespace suffix, user-given *namespace* or generated timestamp string when using *namespace-mangling*

- non-FATE script: ``Main`` always has one input:

  - param: job parameter setting, dictionary loaded from "conf" file specified in benchmark testsuite


data
----

`Data` sub-command is used for upload or delete dataset in suite's.

command options
~~~~~~~~~~~~~~~

.. code-block:: bash

      fate_test data --help

1. include:

   .. code-block:: bash

      fate_test data [upload|delete] -i <path1 contains *testsuite.json>

   will upload/delete dataset in testsuites in *path1*

2. exclude:

   .. code-block:: bash

      fate_test data [upload|delete] -i <path1 contains *testsuite.json> -e <path2 to exclude> -e <path3 to exclude> ...

   will upload/delete dataset in testsuites in *path1* but not in *path2* and *path3*

3. glob:

   .. code-block:: bash

      fate_test data [upload|delete] -i <path1 contains *testsuite.json> -g "hetero*"

   will upload/delete dataset in testsuites in sub directory start with *hetero* of *path1*


full command options
---------------------

.. click:: fate_test.scripts.cli:cli
  :prog: fate_test
  :show-nested:
