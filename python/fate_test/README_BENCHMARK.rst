Benchmark Quality
=================

Benchmark-quality is used for comparing modeling quality between FATE
and other machine learning systems. Benchmark produces a metrics summary
for each benchmark task group.

.. code-block:: bash

    fate_test benchmark-quality -i <path contains *testsuite.json>

::
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

suite command options
----------------------

1. include:

.. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json>

   will run testsuites in `path1`

2. exclude:

   .. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json> -e <path2 to exclude> -e <path3 to exclude> ...

   will run testsuites in `path1` but not in `path2` and `path3`

3. glob:

   .. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json> -g "hetero*"

   will run testsuites in sub directory start with `hetero` of `path1`

