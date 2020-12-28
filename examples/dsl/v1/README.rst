FATE Usage
==========

If you want to experience FATE quickly, we have provided you a quick start tool which can start a hetero-lr task quickly. After that, you are more than welcome to use provided configuration to experience algorithms listed. Before you upload and start training task, it is highly recommended that you read the configuration guide below.

Start Training Task Manually
============================

There are three config files need to be prepared to build a algorithm model in FATE.

1. Upload data config file: for upload data
2. DSL config file: for defining your modeling task
3. Submit runtime conf: for setting parameters for each component


Step1: Define upload data config file
-------------------------------------

To enable FATE to use your data, you need to upload them. Thus, a upload-data conf is needed. A sample file named "upload_data.json" has been provided in current folder.

Field Specification
^^^^^^^^^^^^^^^^^^^

:file: file path
:head: Specify whether your data file include a header or not
:partition: Specify how many partitions used to store the data
:table_name & namespace: Indicators for stored data table.
:work_mode: Indicate if using standalone version or cluster version. 0 represent for standalone version and 1 stand for cluster version.

.. Note::
    We suggest you fully consider the resource of modeling machines before setting partition number.

    Assume that the CPU cores (cpu cores) are: c, The number of Nodemanager is: n, The number of tasks to be run simultaneously is p, then:

    egg_num=eggroll.session.processors.per.node = c * 0.8 / p

    partitions (Number of roll pair partitions) = egg_num * n


Step2: Define your modeling task structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Practically, when building a modeling task, several components might be involved, such as data_io, feature_engineering, algorithm_model, evaluation as so on. However, the combination of these components would differ from task to task. Therefore, a convenient way to freely combine these components would be a critical feature.

Currently, FATE provide a kind of domain-specific language(DSL) to define whatever structure you want. The components are combined as a Directed Acyclic Graph(DAG) through the dsl config file. The usage of dsl config file is as simple as defining a json file.

The DSL config file will define input data and(or) model as well as output data and(or) model for each component. The downstream components take output data and(or) model of upstream components as input. In this way, a DAG can be constructed by the config file.

We have provided several example dsl files located in the corresponding algorithm folder. For example, hetero-lr dsl files are located `here <hetero_logistic_regression/test_hetero_lr_train_job_dsl.json>`_.


Field Specification
^^^^^^^^^^^^^^^^^^^

:component_name: key of a component. This name should end with a "_num" such as "_0", "_1" etc. And the number should start with 0. This is used to distinguish multiple same kind of components that may exist.

:module: Specify which component use. This field should be one of the algorithm modules FATE supported.
         The supported algorithms can be referred to `here <../../federatedml/README.rst>`__

    - input: There are two type of input, data and model.

      - data: There are three possible data_input type:

        * data: typically used in data_io, feature_engineering modules and evaluation.
        * train_data: Used in homo_lr, hetero_lr and secure_boost. If this field is provided, the task will be parse as a **fit** task
        * validate_data: If train_data is provided, this field is optional. In this case, this data will be used as validation set. If train_data is not provided, this task will be parse as a **predict** or **transform** task.

      - model: There are two possible model-input type:

        * model: This is a model input by same type of component, used in prediction or transform stage. For example, hetero_binning_0 run as a fit component, and hetero_binning_1 take model output of hetero_binning_0 as input so that can be used to transform or predict.
        * isometric_model: This is used to specify the model input from upstream components, only used by HeteroFeatureSelection module in FATE-1.x. HeteroFeatureSelection can take the model output of HetereFeatureBinning and use information value calculated as filter criterion.

    - output: Same as input, two type of output may occur which are data and model.

      - data: Specify the output data name
      - model: Specify the output model name

:need_deploy: true or false. This field is used to specify whether the component need to deploy for online inference or not. This field just use for online-inference dsl deduction.


Step3: Define Submit Runtime Configuration for Each Specific Component.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This config file is used to config parameters for all components among every party.

1. initiator: Specify the initiator's role and party id.
2. role: Indicate all the party ids for all roles.
3. role_parameters: Those parameters are differ from roles and roles are defined here separately. Please note each parameter are list, each element of which corresponds to a party in this role.
4. algorithm_parameters: Those parameters are same among all parties are here.

An example of config files can be shown as:

.. code-block::

    {
        "initiator": {
            "role": "guest",
            "party_id": 10000
        },
        "job_parameters": {
            "work_mode": 1
            "processor_per_node": 6
        },
        "role": {
            "guest": [
                10000
            ],
            "host": [
                10000
            ],
            "arbiter": [
                10000
            ]
        },
        "role_parameters": {"Your role parameters"},
        "algorithm_parameters": {"Your algorithm parameters"},
    }

You can set processor_per_node in job_parameters.

Step4: Start Modeling Task
^^^^^^^^^^^^^^^^^^^^^^^^^^

:Upload data:
    Before starting a task, you need to load data among all the data-providers. To do that, a load_file config is needed to be prepared.  Then run the following command:

    .. code-block::

        python ${your_install_path}/fate_flow/fate_flow_client.py -f upload -c upload_data.json

    Here is an example of configuring upload_data.json:

    .. code-block:: json

        {
          "file": "examples/data/breast_hetero_guest.csv",
          "head": 1,
          "partition": 8,
          "work_mode": 0,
          "table_name": "breast_hetero_guest",
          "namespace": "experiment"
        }

    We use **breast_hetero_guest** & **experiment** as guest party's table name and namespace. To use default runtime conf, please set host party's name and namespace as **breast_hetero_host** & **hetero_host_breast** and upload the data with path of  **examples/data/breast_hetero_host.csv**

    To use other data set, please change your file path and table_name & namespace. Please do not upload different data set with same table_name and namespace.

    .. Note::

        This step is needed for every data-provide node(i.e. Guest and Host).

:Start your modeling task:
    In this step, two config files corresponding to dsl config file and submit runtime conf file should be prepared. Please make sure the table_name and namespace in the conf file match with upload_data conf.

    ::

      "role_parameters": {
        "guest": {
            "args": {
                "data": {
                    "train_data": [{"name": "breast_hetero_guest", "namespace": "experiment"}]
                }
            }
 

    As the above example shows, the input train_data should match the upload file conf.

    Then run the following command:

    .. code-block:: bash
        
        python ${your_install_path}/fate_flow/fate_flow_client.py -f submit_job -d hetero_logistic_regression/test_hetero_lr_train_job_dsl.json -c hetero_logistic_regression/test_hetero_lr_train_job_conf.json

:Check log files:
    Now you can check out the log in the following path: ${your_install_path}/logs/{your jobid}.


Step5: Check out Results
^^^^^^^^^^^^^^^^^^^^^^^^

FATE now provide "FATE-BOARD" for showing modeling log-metrics and evaluation results.

Use your browser to open a website: `http://{Your fate-board ip}:{your fate-board port}/index.html#/history`.

.. figure:: ../../image/JobList.png
   :height: 250
   :align: center
   
   Figure 1: Job List

There will be all your job history list here. Your latest job will be list in the first page. Use JOBID to find out the modeling task you want to check.

.. figure:: ../../image/JobOverview.png
   :height: 250
   :align: center
   
   Figure 2: Job Overview

In the task page, all the components will be shown as a DAG. We use different color to indicate their running status.

1. Green: run success
2. Blue: running
3. Gray: Waiting
4. Red: Failed.

You can click each component to get their running parameters on the right side. Below those parameters, there exist a **View the outputs** button. You may check out model output, data output and logs for this component.

.. figure:: ../../image/Component_Output.png
   :height: 250
   :align: center
   
   Figure 3: Component Output

If you want a big picture of the whole task, there is a **dashboard** button on the right upper corner. Get in the Dashboard, there list three windows showing different information.

.. figure:: ../../image/DashBoard.png
   :height: 250
   :align: center
   
   Figure 4: Dash Board


1. Left window: showing data set used for each party in this task.
2. Middle window: Running status or progress of the whole task.
3. Right window: DAG of components.


Step6: Check out Logs
^^^^^^^^^^^^^^^^^^^^^

After you submit a job, you can find your job log in `${Your install path}/logs/${your jobid}`

The logs for each party is collected separately and list in different folders. Inside each folder, the logs for different components are also arranged in different folders. In this way, you can check out the log more specifically and get useful detailed  information.


FATE-FLOW Usage
---------------

How to get the output data of each component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd {your_fate_path}/fate_flow

   python fate_flow_client.py -f component_output_data -j $jobid -p $party_id -r $role -cpn $component_name -o $output_dir


:jobid: the task jobid you want to get.

:party_id: your mechine's party_id, such as 10000

:role: "guest" or "host" or "arbiter"
 
:component_name: the component name which you want to get, such as component_name "hetero_lr_0" in 
   ::
      
      {your_fate_path}/examples/dsl/v1/hetero_logistic_regression/test_hetero_lr_train_job_dsl.json

:output_dir: the output directory


How to get the output model of each component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
.. code-block:: bash
   
   python fate_flow_client.py -f component_output_model -j $jobid -p $party_id -r $role -cpn $component_name

How to get the logs of task
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash
   
   python fate_flow_client.py -f job_log -j $jobid -o $output_dir
 
How to stop the job
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash
   
   python fate_flow_client.py -f stop_job -j $jobid

How to query job current status
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python fate_flow_client.py -f query_job -j $jobid -p party_id -r role

How to get the job runtime configure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash
   
   python fate_flow_client.py -f job_config -j $jobid -p party_id -r role -o $output_dir

How to download a table which has been uploaded before
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash
   
   python fate_flow_client.py -f download -n table_namespace -t table_name -w work_mode -o save_file
 
:work_mode: will be 0 for standalone or 1 for cluster, which depend on what you set in upload config


Predict Task Usage
------------------

In order to use trained model to predict. The following several steps are needed.

Step1: Train Model
^^^^^^^^^^^^^^^^^^

Pay attention to following points to enable predicting:

1. you should add or modify "need_deploy" field for those modules that need to deploy in predict stage. All modules have set True as their default value except FederatedmSample and Evaluation, which typically will not run in predict stage. The "need_deploy" field is True means this module should run a "fit" process and the fitted model need to be deployed in predict stage.

2. Besides setting those model as "need_deploy", they should also config to have a model output except Intersect module. Only in this way can fate-flow store the trained model and make it usable in inference stage.

3. Get training model's model_id and model_version. There are two ways to get this.

   a. After submit a job, there will be some model information output in which "model_id" and "model_version" are our interested field.

   b. Besides that, you can also obtain these information through the following command directly:

      .. code-block:: bash
          
         python ${your_fate_install_path}/fate_flow/fate_flow_client.py -f job_config -j ${jobid} -r guest -p ${guest_partyid}  -o ${job_config_output_path}
       
      where

      :guest_partyid: the partyid of guest (the party submitted the job)
      :job_config_output_path: path to store the job_config

      After that, a json file including model info will be download to ${job_config_output_path}/model_info.json in which you can find "model_id" and "model_version".


Step2: define your predict config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This config file is used to config parameters for predicting.

1. initiator: Specify the initiator's role and party id, it should be same with training process.
2. job_parameters:

   - work_mode: cluster or standalone, it should be same with training process.
   - model_id or model_version: model indicator which mentioned in Step1.
   - job_type: type of job. In this case, it should be "predict".

   There is an example test `config file <./test_predict_conf.json>`_
3. role: Indicate all the party ids for all roles, it should be same with training process.
4. role_parameters: Set parameters for each roles. In this case, the "validate_data", which means data going to be predicted, should be filled for both Guest and Host parties.

Step3. Start your predict task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After complete your predict configuration, run the following command.

.. code-block:: bash
   
   python ${your_fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${predict_config}

Step4: Check out Running State
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running State can be check out in FATE_board whose url is 
::

   http://${fate_board_ip}:${fate_board_port}/index.html#/details?job_id=${job_id}&role=guest&party_id=${guest_partyid}

where

- ${fate_board_ip}\${fate_board_port}: ip and port to deploy the FATE board module.

- ${job_id}: the predict task's job_id.

- ${guest_partyid}: the guest party id

You can also checkout job status by fate_flow in case without FATE_board installed. The following command is used to query job status such as running, success or fail.

.. code-block:: bash
   
   python ${your_fate_install_path}/fate_flow/fate_flow_client.py -f query_job -j {job_id} -r guest


Step5: Download Predicting Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once predict task finished, the first 100 records of predict result are available in FATE-board. You can also download all results through the following command.

.. code-block:: bash
  
  python ${your_fate_install_path}/fate_flow/fate_flow_client.py -f component_output_data -j ${job_id} -p ${party_id} -r ${role} -cpn ${component_name} -o ${predict_result_output_dir}

where

- ${job_id}: predict task's job_id
- ${party_id}: the partyid of current user.
- ${role}: the role of current user. Please keep in mind that host users are not supposed to get predict results in heterogeneous algorithm.
- ${component_name}: the component who has predict results
- ${predict_result_output_dir}: the directory which use download the predict result to.


use spark
---------

1. deploy spark(yarn or standalone)
2. export SPARK_HOME env before fate_flow service start(better adding env to service.sh)
3. adjust runtime_conf, adjust job_parameters field:
   
   .. code-block:: json

      {
        "job_parameters": {
            "backend": 1,
            "spark_submit_config": {
                "deploy-mode": "client",
                "queue": "default",
                "driver-memory": "1g",
                "num-executors": 2,
                "executor-memory": "1g",
                "executor-cores": 1
            }
        }
      }
