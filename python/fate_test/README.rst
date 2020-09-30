FATE Test
=========

A collection of useful tools to running FATE's test.

quick start
-----------

1. (optional) create virtual env

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate


2. install fate_test

   .. code-block:: bash

      pip install fate_test
      fate_test --help


3. new and edit the fate_test_config.yaml

   .. code-block:: bash

      # create a fate_test_config.yaml in current dir
      fate_test config new
      # edit priority config file with system default editor
      # filling some field according to comments
      fate_test config edit


4. run some fate_test suite

   .. code-block:: bash

      fate_test suite -i <path contains *testsuite.json>


5. run some fate_test benchmark

   .. code-block:: bash

      fate_test benchmark-quality -i <path contains *testsuite.json>

6. useful logs or exception will be saved to logs dir with namespace showed in last step



command types
-------------

- suite: used for running testsuites, collection of FATE jobs

  .. code-block:: bash

     fate_test suite -i <path contains *testsuite.json>


- benchmark-quality used for comparing modeling quality between FATE and other machine learning systems

  .. code-block:: bash

      fate_test benchmark-quality -i <path contains *testsuite.json>




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

Benchmark Quality
------------------

Benchmark-quality is used for comparing modeling quality between FATE
and other machine learning systems. Benchmark produces a metrics comparison
summary for each benchmark job group.

.. code-block:: bash

   fate_test benchmark-quality -i hetero_linr_sklearn_benchmark.json

output comparison summary ::

    +------------+--------------------+---------------------+--------------------+-------------------------+
    | Model Name |      r2_score      |  mean_squared_error | explained_variance | root_mean_squared_error |
    +------------+--------------------+---------------------+--------------------+-------------------------+
    |   local    | 0.8996802446941182 |  0.1021175724655836 | 0.899680245220208  |    0.3195584022766161   |
    |  pipeline  | 0.9025618809878748 | 0.09918429474625605 | 0.9026740215636323 |    0.3149353818583362   |
    +------------+--------------------+---------------------+--------------------+-------------------------+
    +-------------------------+-----------+
    |          Metric         | All Match |
    +-------------------------+-----------+
    |         r2_score        |    True   |
    |    mean_squared_error   |    True   |
    |    explained_variance   |    True   |
    | root_mean_squared_error |    True   |
    +-------------------------+-----------+

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

4. config:

   .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> -c <path2 to *.yaml>

   will run benchmark testsuites in *path1* with config file at *path2*

5. tol:

   .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> -t 1e-3

   will run benchmark testsuites in *path1* with absolute tolerance of difference between metrics set to 0.001.
   If absolute difference between metrics is smaller than *tol*, then metrics are considered
   almost equal. Check benchmark testsuite `writing guide <#benchmark-testsuite>`_ on setting alternative tolerance.

6. data-namespace-mangling:

   .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> --data-namespace-mangling

   will run benchmark testsuites in *path1* with uploaded data namespace modified to have a suffix of timestamp.
   Timestamp is used for distinguishing data from different tetsuites.
   Uploaded data will be deleted after all benchmark jobs end.

7. skip-data:

   .. code-block:: bash

       fate_test benchmark-quality -i <path1 contains *benchmark.json> --skip-date

   will run benchmark testsuites in *path1* without uploading data specified in *benchmark.json*.
   Note that data-namespace-mangling is ineffective when skipping data upload.

8. yes:

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
                "file": "../../data/motor_hetero_host.csv",
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
            "script": "./sklearn-linr.py",
            "conf": "./linr_config.yaml"
       }

  - compare_setting: additional setting for quality metrics comparison, currently only takes ``relative_tol``

    If metrics *a* and *b* satisfy *abs(a-b) <= max(relative_tol \* max(abs(a), abs(b)), absolute_tol)*
    (from `math module <https://docs.python.org/3/library/math.html#math.isclose>`_),
    they are considered almost equal. In the below example, metrics from "local" and "pipeline" jobs are
    considered almost equal if their relative difference is smaller than
    *0.05 \* max(abs(local_metric), abs(pipeline_metric)*.

  .. code-block:: json

     "binary": {
         "local": {
             "script": "./sklearn-linr.py",
             "conf": "./linr_config.yaml"
         },
         "pipeline": {
             "script": "./pipeline-linr.py",
             "conf": "./linr_config.yaml"
         },
         "compare_setting": {
             "relative_tol": 0.05
         }
     }


testing script
~~~~~~~~~~~~~~

All job scripts need to have ``Main`` function as an entry point for executing jobs; scripts should
return a dictionary with {metric_name}: {metric_value} key-value pairs for comparison.
Returned quality metrics of the same key are to be compared.
Note that only **real-value** metrics can be compared.

- FATE script: ``Main`` always has three inputs:

  - config: job configuration, `JobConfig <../fate_client/pipeline/utils/tools.py#L64>`_ object loaded from "fate_test_config.yaml"
  - param: job parameter setting, dictionary loaded from "conf" file specified in benchmark testsuite
  - namespace: namespace suffix, generated timestamp string when using *data-namespace-mangling*

- non-FATE script: ``Main`` always has one input:

  - param: job parameter setting, dictionary loaded from "conf" file specified in benchmark testsuite



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

      fate_test suite -i <path1 contains *benchmark.json>

   will run testsuites in *path1*

2. exclude:

   .. code-block:: bash

      fate_test suite -i <path1 contains *benchmark.json> -e <path2 to exclude> -e <path3 to exclude> ...

   will run testsuites in *path1* but not in *path2* and *path3*

3. glob:

   .. code-block:: bash

      fate_test suite -i <path1 contains *benchmark.json> -g "hetero*"

   will run testsuites in sub directory start with *hetero* of *path1*

4. config:

   .. code-block:: bash

      fate_test suite -i <path1 contains *benchmark.json> -c <path2 to *.yaml>

   will run testsuites in *path1* with config file at *path2*

5. tol:

   .. code-block:: bash

      fate_test suite -i <path1 contains *benchmark.json> -t 1e-3

   will run testsuites in *path1* with absolute tolerance of difference between metrics set to 0.001.
   If absolute difference between metrics is smaller than *tol*, then metrics are considered
   almost equal. Check benchmark testsuite `writing guide <#benchmark-testsuite>`_ on setting alternative tolerance.

6. data-namespace-mangling:

   .. code-block:: bash

      fate_test suite -i <path1 contains *benchmark.json> --data-namespace-mangling

   will run testsuites in *path1* with uploaded data namespace modified to have a suffix of timestamp.
   Timestamp is used for distinguishing data from different tetsuites.
   Uploaded data will be deleted after all benchmark jobs end.

7. skip-data:

   .. code-block:: bash

       fate_test suite -i <path1 contains *benchmark.json> --skip-date

   will run testsuites in *path1* without uploading data specified in *benchmark.json*.
   Note that data-namespace-mangling is ineffective when skipping data upload.

8. yes:

   .. code-block:: bash

      fate_test suite -i <path1 contains *benchmark.json> --yes

   will run testsuites in *path1* directly, skipping double check


full command options
---------------------

.. click:: fate_test.cli:cli
  :prog: fate_test
  :show-nested:
