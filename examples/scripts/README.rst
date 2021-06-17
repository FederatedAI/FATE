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

     python upload_default_data.py -m {work_mode}

Where work_mode stands for whether you are running standalone mode or cluster mode. If it is standalone mode, work_mode is 0 while work_mode equals to 1 for cluster mode.

After that, you are expected to see the following feedback which showing the table information for you:

    ::

    [2020-06-12 14:19:39]uploading @examples/data/breast_hetero_guest.csv >> experiment.breast_hetero_guest
    [2020-06-12 14:19:39]upload done @examples/data/breast_hetero_guest.csv >> experiment.breast_hetero_guest, job_id=2020061214193960279930

    [2020-06-12 14:19:42]2020061214193960279930 success, elapse: 0:00:02
    [2020-06-12 14:19:42] check_data_out {'data': {'count': 569, 'namespace': 'experiment', 'partition': 16, 'table_name': 'breast_hetero_guest'}, 'retcode': 0, 'retmsg': 'success'}

    [2020-06-12 14:19:42]uploading @examples/data/breast_hetero_host.csv >> experiment.breast_hetero_host
    [2020-06-12 14:19:43]upload done @examples/data/breast_hetero_host.csv >> experiment.breast_hetero_host, job_id=2020061214194343127831

    [2020-06-12 14:19:46]2020061214194343127831 success, elapse: 0:00:02
    [2020-06-12 14:19:46] check_data_out {'data': {'count': 569, 'namespace': 'experiment', 'partition': 16, 'table_name': 'breast_hetero_host'}, 'retcode': 0, 'retmsg': 'success'}

    [2020-06-12 14:19:46]uploading @examples/data/default_credit_hetero_guest.csv >> experiment.default_credit_hetero_guest
    [2020-06-12 14:19:47]upload done @examples/data/default_credit_hetero_guest.csv >> experiment.default_credit_hetero_guest, job_id=2020061214194729360032

    [2020-06-12 14:19:50]2020061214194729360032 success, elapse: 0:00:02
    [2020-06-12 14:19:50] check_data_out {'data': {'count': 30000, 'namespace': 'experiment', 'partition': 16, 'table_name': 'default_credit_hetero_guest'}, 'retcode': 0, 'retmsg': 'success'}

    [2020-06-12 14:19:50]uploading @examples/data/default_credit_hetero_host.csv >> experiment.default_credit_hetero_host
    [2020-06-12 14:19:51]upload done @examples/data/default_credit_hetero_host.csv >> experiment.default_credit_hetero_host, job_id=2020061214195123380833

    [2020-06-12 14:19:54]2020061214195123380833 success, elapse: 0:00:02
    [2020-06-12 14:19:54] check_data_out {'data': {'count': 30000, 'namespace': 'experiment', 'partition': 16, 'table_name': 'default_credit_hetero_host'}, 'retcode': 0, 'retmsg': 'success'}

If you want to set the data you want to upload, please give the configuration file in the following format.

    .. code-block:: json

        upload_data = {
            "data": [
                {
                    "file": "examples/data/breast_hetero_guest.csv",
                    "head": 1,
                    "partition": 16,
                    "table_name": "breast_hetero_guest",
                    "namespace": "experiment",
                    "count": 569
                },
                {
                    "file": "examples/data/breast_hetero_host.csv",
                    "head": 1,
                    "partition": 16,
                    "table_name": "breast_hetero_host",
                    "namespace": "experiment",
                    "count": 569
                },
                {
                    "file": "examples/data/default_credit_hetero_guest.csv",
                    "head": 1,
                    "partition": 16,
                    "table_name": "default_credit_hetero_guest",
                    "namespace": "experiment",
                    "count": 30000
                },
                {
                    "file": "examples/data/default_credit_hetero_host.csv",
                    "head": 1,
                    "partition": 16,
                    "table_name": "default_credit_hetero_host",
                    "namespace": "experiment",
                    "count": 30000
                }
            ]
        }


Parameters
``````````
-  -m --mode: Work mode, Required. 1 represent for cluster version while 0 means standalone version.
-  -f --force: Whether force upload or not. When setting as 1, the table will be deleted before upload if it is already existed. Default: 0
-  -b --backend: Backend of the task. 0 represent for eggroll while 1 represent for spark with rabbitmq and 2 stands for spark with pulsar. Default: 0
-  -c --config_file: The config file provided. If a file path is provided, it will upload the data list in the config file. We also provided some pre-set config file. Default: min-test
    *  "all" means upload all data-set provided in example/data folder. If use this config file, the time consume for this upload task could be relatively long.
    *  "min-test" means upload the data needed for min-test.

An example of starting this script with all parameter could be:

  .. code-block:: bash

     python upload_default_data.py -m 0 -f 1 -b 0 -c min-test


Make Conf & DSL from Pipeline file
----------------------------------

If you already have a pipeline py file and want to generate conf & dsl files, this tool would be a good helper. Please make sure your pipeline file have a "main" function and a "pipeline" variable. This script will make conf based on the "pipeline" variable in main function.

To use it, the command is as simple as:

  .. code-block:: bash

     python make_conf_dsl.py -c ${your pipeline file}