Hetero Kmeans Configuration Usage Guide.
----------------------------------------

This section introduces the dsl and conf for usage of different type of
task.

Upload data
^^^^^^^^^^^

We have provided several upload config for you can upload example data
conveniently.

1. breast data set

   1. Guest Party Data: upload\_data\_guest.json
   2. Host Party Data: upload\_data\_host.json

   This data set can be applied for train task, train & validation task
   and with feature engineering task that list below.

2. vehicle data set

   1. Guest Party Data: upload\_vehicle\_guest.json
   2. Host Party Data: upload\_vehicle\_host.json

   This data set can be applied for multi-class task.

Training Task.
^^^^^^^^^^^^^^

1. Train\_task: dsl: test\_hetero\_kmeans\_job\_dsl.json runtime\_config
   : test\_hetero\_kmeans\_job\_conf.json

2. Train, test and evaluation task: dsl:
   test\_hetero\_kmeans\_validate\_job\_dsl.json runtime\_config:
   test\_hetero\_kmeans\_validate\_job\_conf.json

3. LR with feature engineering task dsl:
   test\_hetero\_kmeans\_with\_feature\_engineering\_dsl.json conf:
   test\_hetero\_kmeans\_with\_feature\_engineering\_job\_conf.json

4. Multi-host training task: dsl: test\_hetero\_kmeans\_job\_dsl.json
   conf: test\_hetero\_kmeans\_job\_multi\_host\_conf.json

5. Predict task: "conf": "test\_predict\_conf.json"

Users can use following commands to running the task.

::

    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it
to predict too.
