Description
===========

Here we place some scripts with auxiliary functions.

Upload Data
-----------

We have already placed some public data sets in `examples/data <../data>`_. We also provide a script to upload data into FATE through FATE-Flow interface.

Description
```````````

You can use it as simple as running the following command:

  .. code-block:: bash

     python upload_default_data.py

After that, you are expected to see the following feedback which showing the table information for you:

    ::

    [2020-06-12 14:19:39]uploading @examples/data/breast_hetero_guest.csv >> experiment.breast_hetero_guest
    [2020-06-12 14:19:39]upload done @examples/data/breast_hetero_guest.csv >> experiment.breast_hetero_guest, job_id=2020061214193960279930
    [2020-06-12 14:19:42]2020061214193960279930 success, elapse: 0:00:02
    [2020-06-12 14:19:42] check_data_out {'data': {'count': 569, 'namespace': 'experiment', 'partition': 16, 'table_name': 'breast_hetero_guest'}, 'retcode': 0, 'retmsg': 'success'}


If you want to set the data you want to upload, please give the configuration file in the following format : `min_test data config <./min_test_config.json>`__


Parameters
``````````
-  -f --force: Whether force upload or not. When setting as 1, the table will be deleted before upload if it is already existed. Default: 0
-  -c --config_file: The config file provided. If a file path is provided, it will upload the data list in the config file. We also provided some pre-set config file. Default: min-test
    *  "all" means upload all data-set provided in example/data folder. If use this config file, the time consume for this upload task could be relatively long.
    *  "min-test" means upload the data needed for min-test.

An example of starting this script with all parameter could be:

  .. code-block:: bash

     python upload_default_data.py -f 1 -c min-test


Make Conf & DSL from Pipeline file
----------------------------------

If you already have a pipeline py file and want to generate conf & dsl files, this tool would be a good helper. Please make sure your pipeline file have a "main" function and a "pipeline" variable. This script will make conf based on the "pipeline" variable in main function.

To use it, the command is as simple as:

  .. code-block:: bash

     python make_conf_dsl.py -c ${your pipeline file}