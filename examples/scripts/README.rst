Description
===========

Here we placed some scripts with auxiliary functions.

Upload Data
-----------

We have already placed some public data sets in example/data. We also provide a script to upload these data into FATE through fate-flow interface.

You can use it as simple as running the following command:

  .. code-block:: bash

     python upload_default_data.py -m {work_mode}

Where work_mode stands for whether you are running standalone mode or cluster mode. If it is standalone mode, work_mode is 0 while work_mode equal to 1 stands for cluster mode.

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

If you want to use this script to upload more data set, this could be set at the beginning of this script:

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

If these table name and namespace has been used and you want to delete them before upload, you can just add one parameter to achieve that.

  .. code-block:: bash

     python upload_default_data.py -m {work_mode} -f 1

