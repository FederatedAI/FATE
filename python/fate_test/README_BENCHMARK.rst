Benchmark Quality
=================

Benchmark-quality is used for comparing modeling quality between FATE
and other machine learning systems. Benchmark produces a metrics summary
for each benchmark task group.

.. code-block:: bash

    fate_test benchmark-quality -i hetero_linr_sklearn_benchmark.json

output::

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
---------------

use the following command to show help message

.. code-block:: bash

      fate_test benchmark-quality --help

1. include:

    .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json>

   will run benchmark testsuites in `path1`

2. exclude:

   .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> -e <path2 to exclude> -e <path3 to exclude> ...

   will run benchmark testsuites in `path1` but not in `path2` and `path3`

3. glob:

   .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> -g "hetero*"

   will run benchmark testsuites in sub directory start with `hetero` of `path1`

4. config:

   .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> -c <path2 to *.yaml>

   will run benchmark testsuites in `path1` with config file at `path2`

5. tol:

    .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> -t 1e-3

   will run benchmark testsuites in `path1` with absolute tolerance of difference between metrics set to 0.001.
   If absolute difference between metrics is smaller than `tol`, then metrics are considered
   almost equal. Check testing conf `writing guide <#testing conf>`_ on setting alternative tolerance.

6. data-namespace-mangling:

    .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> --data-namespace-mangling

    will run benchmark testsuites in `path1` with uploaded data namespace modified to have a suffix of timestamp.
    Timestamp is used for distinguishing data from different tetsuites.
    Uploaded data will be deleted after all benchmark jobs end.

7. skip-data

    .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> --skip-date

    will run benchmark testsuites in `path1` without uploading data specified in `*benchmark.json`.
    Note that data-namespace-mangling is ineffective when skipping data upload.

8. yes

    .. code-block:: bash

      fate_test benchmark-quality -i <path1 contains *benchmark.json> --yes

    will run benchmark testsuites in `path1` directly, skipping double check


testing conf
------------

Configuration of jobs need to be specified in a json file with ending "*benchmark.json".
A benchmark testsuite file should include the following elements:


testing script
--------------
