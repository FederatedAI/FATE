FATE-Flow Client Command Line Interface v2 Guide
================================================

[`中文`_]

.. _中文: README_zh.rst

Usage
-----

Before using fate flow client command line interface (CLI), please make
sure that you have activated the virtual environment of FATE. For more
details about how to activate virtual environment, please read the
documentation of deployment.

In this version of client CLI, commands are separated into several
classes, including *job*, *data*, *model*, *component* and etc. And all
of these classes have a common parent (CLI entry) named *'flow'*, which
means you can type *'flow'* in your terminal window to find out all of
these classes and also their sub-commands.

.. code:: bash

    [IN]
    flow

    [OUT]
    Usage: flow [OPTIONS] COMMAND [ARGS]...

      Fate Flow Client

    Options:
      -h, --help  Show this message and exit.

    Commands:
      component   Component Operations
      data        Data Operations
      job         Job Operations
      model       Model Operations
      queue       Queue Operations
      table       Table Operations
      task        Task Operations

For more details, please check this documentation or try ``flow --help``
for help.

Init
----

``init``
~~~~~~~~

-  *Description*: Flow CLI Init Command. Custom can choose to provide an
   absolute path of server conf file, or provide ip address and http
   port of a valid fate flow server. Notice that, if custom provides
   both, the server conf would be loaded in priority. In this case, ip
   address and http port would be ignored.
-  *Arguments*:

+-------+--------------+-----------+--------------------------+------------+-----------------------------------------------------------------------------+
| No.   | Argument     | Flag\_1   | Flag\_2                  | Required   | Description                                                                 |
+=======+==============+===========+==========================+============+=============================================================================+
| 1     | conf\_path   | ``-c``    | ``--server-conf-path``   | No         | Server configuration file absolute path                                     |
+-------+--------------+-----------+--------------------------+------------+-----------------------------------------------------------------------------+
| 2     | ip           |           | ``--ip``                 | No         | Fate flow server ip address                                                 |
+-------+--------------+-----------+--------------------------+------------+-----------------------------------------------------------------------------+
| 3     | port         |           | ``--port``               | No         | Fate flow server port                                                       |
+-------+--------------+-----------+--------------------------+------------+-----------------------------------------------------------------------------+
| 4     | reset        |           | ``--reset``              | No         | If specified, initialization settings of flow CLI would be reset to none.   |
+-------+--------------+-----------+--------------------------+------------+-----------------------------------------------------------------------------+

-  *Examples*:

.. code:: bash

    flow init -c /data/projects/fate/python/conf/service_conf.yaml
    flow init --ip 127.0.0.1 --port 9380

Job
---

``submit``
~~~~~~~~~~

-  *Description*: Submit a pipeline job.
-  *Arguments*:

+-------+--------------+-----------+-------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| No.   | Argument     | Flag\_1   | Flag\_2           | Required   | Description                                                                                                                                                                                                |
+=======+==============+===========+===================+============+============================================================================================================================================================================================================+
| 1     | conf\_path   | ``-c``    | ``--conf-path``   | Yes        | Runtime configuration file path                                                                                                                                                                            |
+-------+--------------+-----------+-------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 2     | dsl\_path    | ``-d``    | ``--dsl-path``    | Yes        | Domain-specific language(DSL) file path. If the type of job is 'predict', you can leave this feature blank, or you can provide a valid dsl file to replace the one that aotumatically generated by fate.   |
+-------+--------------+-----------+-------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

-  *Examples*:

.. code:: bash

    flow job submit -c fate_flow/examples/test_hetero_lr_job_conf.json -d fate_flow/examples/test_hetero_lr_job_dsl.json

``stop``
~~~~~~~~

-  *Description*: Cancel or stop a specified job.
-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+-------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description       |
+=======+===================+============+========================+============+===================+
| 1     | job\_id           | ``-j``     | ``--job_id``           | Yes        | A valid job id.   |
+-------+-------------------+------------+------------------------+------------+-------------------+

-  *Examples*:

   .. code:: bash

       flow job stop -j $JOB_ID

``query``
~~~~~~~~~

-  *Description*: Query job information by filters.
-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+-------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description       |
+=======+===================+============+========================+============+===================+
| 1     | job\_id           | ``-j``     | ``--job_id``           | No         | A valid job id.   |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 2     | role              | ``-r``     | ``--role``             | No         | Role              |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 3     | party\_id         | ``-p``     | ``--party_id``         | No         | Party ID          |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 4     | status            | ``-s``     | ``--status``           | No         | Job Status        |
+-------+-------------------+------------+------------------------+------------+-------------------+

-  *Examples*:

   .. code:: bash

       flow job query -r guest -p 9999 -s complete
       flow job query -j $JOB_ID


``view``
~~~~~~~~

-  *Description*: Query data view information by filters.

-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+-------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description       |
+=======+===================+============+========================+============+===================+
| 1     | job\_id           | ``-j``     | ``--job_id``           | Yes         | A valid job id.   |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 2     | role              | ``-r``     | ``--role``             | No         | Role              |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 3     | party\_id         | ``-p``     | ``--party_id``         | No         | Party ID          |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 4     | status            | ``-s``     | ``--status``           | No         | Job Status        |
+-------+-------------------+------------+------------------------+------------+-------------------+

-  *Examples*:

   .. code:: bash

       flow job view -j $JOB_ID -s complete

``config``
~~~~~~~~~~

-  *Description*: Download the configuration of a specified job.
-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+-------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description       |
+=======+===================+============+========================+============+===================+
| 1     | job\_id           | ``-j``     | ``--job_id``           | Yes        | A valid job id.   |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 2     | role              | ``-r``     | ``--role``             | Yes        | Role              |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 3     | party\_id         | ``-p``     | ``--party_id``         | Yes        | Party ID          |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 4     | output\_path      | ``-o``     | ``--output-path``      | Yes        | Output Path       |
+-------+-------------------+------------+------------------------+------------+-------------------+

-  *Examples*\ ：

   .. code:: bash

       flow job config -j $JOB_ID -r host -p 10000 --output-path ./examples/

``log``
~~~~~~~

-  *Description*: Download log files of a specified job.
-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+-------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description       |
+=======+===================+============+========================+============+===================+
| 1     | job\_id           | ``-j``     | ``--job_id``           | Yes        | A valid job id.   |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 2     | output\_path      | ``-o``     | ``--output-path``      | Yes        | Output Path       |
+-------+-------------------+------------+------------------------+------------+-------------------+

-  *Examples*:

   .. code:: bash

       flow job log -j JOB_ID --output-path ./examples/

``list``
~~~~~~~~

-  *Description*: List jobs.
-  *Arguments*:

+-------+------------+-----------+---------------+------------+----------------------------------------------+
| No.   | Argument   | Flag\_1   | Flag\_2       | Required   | Description                                  |
+=======+============+===========+===============+============+==============================================+
| 1     | limit      | ``-l``    | ``--limit``   | No         | Number of records to return. (default: 10)   |
+-------+------------+-----------+---------------+------------+----------------------------------------------+

-  *Examples*:

.. code:: bash

    flow job list
    flow job list -l 30

``dsl``
~~~~~~~

-  *Description*: A predict dsl generator.
-  *Arguments*:

+-------+--------------------+-----------+------------------------+------------+----------------------------------------------------------------+
| No.   | Argument           | Flag\_1   | Flag\_2                | Required   | Description                                                    |
+=======+====================+===========+========================+============+================================================================+
| 1     | cpn\_list          |           | ``--cpn-list``         | No         | User inputs a string to specify component list.                |
+-------+--------------------+-----------+------------------------+------------+----------------------------------------------------------------+
| 2     | cpn\_path          |           | ``--cpn-path``         | No         | User specifies a file path which records the component list.   |
+-------+--------------------+-----------+------------------------+------------+----------------------------------------------------------------+
| 3     | train\_dsl\_path   |           | ``--train-dsl-path``   | Yes        | User specifies the train dsl file path.                        |
+-------+--------------------+-----------+------------------------+------------+----------------------------------------------------------------+
| 4     | output\_path       | ``-o``    | ``--output-path``      | No         | User specifies output directory path.                          |
+-------+--------------------+-----------+------------------------+------------+----------------------------------------------------------------+

-  *Examples*:

.. code:: bash

    flow job dsl --cpn-path fate_flow/examples/component_list.txt --train-dsl-path fate_flow/examples/test_hetero_lr_job_dsl.json

    flow job dsl --cpn-path fate_flow/examples/component_list.txt --train-dsl-path fate_flow/examples/test_hetero_lr_job_dsl.json -o fate_flow/examples/

    flow job dsl --cpn-list "dataio_0, hetero_feature_binning_0, hetero_feature_selection_0, evaluation_0" --train-dsl-path fate_flow/examples/test_hetero_lr_job_dsl.json -o fate_flow/examples/
            
    flow job dsl --cpn-list [dataio_0,hetero_feature_binning_0,hetero_feature_selection_0,evaluation_0] --train-dsl-path fate_flow/examples/test_hetero_lr_job_dsl.json -o fate_flow/examples/

Component (TRACKING)
--------------------

``parameters``
~~~~~~~~~~~~~~

-  *Description*: Query the arguments of a specified component.
-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+-------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description       |
+=======+===================+============+========================+============+===================+
| 1     | job\_id           | ``-j``     | ``--job_id``           | Yes        | A valid job id.   |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 2     | role              | ``-r``     | ``--role``             | Yes        | Role              |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 3     | party\_id         | ``-p``     | ``--party_id``         | Yes        | Party ID          |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 4     | component\_name   | ``-cpn``   | ``--component_name``   | Yes        | Component Name    |
+-------+-------------------+------------+------------------------+------------+-------------------+

-  *Examples*:

.. code:: bash

    flow component parameters -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0

``metric-all``
~~~~~~~~~~~~~~

-  *Description*: Query all metric data.
-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+-------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description       |
+=======+===================+============+========================+============+===================+
| 1     | job\_id           | ``-j``     | ``--job_id``           | Yes        | A valid job id.   |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 2     | role              | ``-r``     | ``--role``             | Yes        | Role              |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 3     | party\_id         | ``-p``     | ``--party_id``         | Yes        | Party ID          |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 4     | component\_name   | ``-cpn``   | ``--component_name``   | Yes        | Component Name    |
+-------+-------------------+------------+------------------------+------------+-------------------+

-  *Examples*:

   .. code:: bash

       flow component metric-all -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0

``metrics``
~~~~~~~~~~~

-  *Description*: Query the list of metrics.
-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+-------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description       |
+=======+===================+============+========================+============+===================+
| 1     | job\_id           | ``-j``     | ``--job_id``           | Yes        | A valid job id.   |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 2     | role              | ``-r``     | ``--role``             | Yes        | Role              |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 3     | party\_id         | ``-p``     | ``--party_id``         | Yes        | Party ID          |
+-------+-------------------+------------+------------------------+------------+-------------------+
| 4     | component\_name   | ``-cpn``   | ``--component_name``   | Yes        | Component Name    |
+-------+-------------------+------------+------------------------+------------+-------------------+

-  *Examples*:

   .. code:: bash

       flow component metrics -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0

``metric-delete``
~~~~~~~~~~~~~~~~~

-  *Description*: Delete specified metric.
-  *Arguments*:

+-------+------------+-----------+----------------+------------+-------------------------------------------------+
| No.   | Argument   | Flag\_1   | Flag\_2        | Required   | Description                                     |
+=======+============+===========+================+============+=================================================+
| 1     | date       | ``-d``    | ``--date``     | No         | An 8-Digit Valid Date, Format Like 'YYYYMMDD'   |
+-------+------------+-----------+----------------+------------+-------------------------------------------------+
| 2     | job\_id    | ``-j``    | ``--job_id``   | No         | Job ID                                          |
+-------+------------+-----------+----------------+------------+-------------------------------------------------+

-  *Examples*:

.. code:: bash

    # NOTICE: If you input both two optional arguments, the 'date' argument will be detected in priority while the 'job_id' argument would be ignored.
    flow component metric-delete -d 20200101
    flow component metric-delete -j $JOB_ID

``output-model``
~~~~~~~~~~~~~~~~

-  *Description*: Query a specified component model.
-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description      |
+=======+===================+============+========================+============+==================+
| 1     | job\_id           | ``-j``     | ``--job_id``           | Yes        | Job ID           |
+-------+-------------------+------------+------------------------+------------+------------------+
| 2     | role              | ``-r``     | ``--role``             | Yes        | Role             |
+-------+-------------------+------------+------------------------+------------+------------------+
| 3     | party\_id         | ``-p``     | ``--party_id``         | Yes        | Party ID         |
+-------+-------------------+------------+------------------------+------------+------------------+
| 4     | component\_name   | ``-cpn``   | ``--component_name``   | Yes        | Component Name   |
+-------+-------------------+------------+------------------------+------------+------------------+

-  *Examples*:

   .. code:: bash

       flow component output-model -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0

``output-data``
~~~~~~~~~~~~~~~

-  *Description*: Download the output data of a specified component.
-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+---------------------------------------------------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description                                                   |
+=======+===================+============+========================+============+===============================================================+
| 1     | job\_id           | ``-j``     | ``--job_id``           | Yes        | Job ID                                                        |
+-------+-------------------+------------+------------------------+------------+---------------------------------------------------------------+
| 2     | role              | ``-r``     | ``--role``             | Yes        | Role                                                          |
+-------+-------------------+------------+------------------------+------------+---------------------------------------------------------------+
| 3     | party\_id         | ``-p``     | ``--party_id``         | Yes        | Party ID                                                      |
+-------+-------------------+------------+------------------------+------------+---------------------------------------------------------------+
| 4     | component\_name   | ``-cpn``   | ``--component_name``   | Yes        | Component Name                                                |
+-------+-------------------+------------+------------------------+------------+---------------------------------------------------------------+
| 5     | output\_path      | ``-o``     | ``--output-path``      | Yes        | User specifies output directory path                          |
+-------+-------------------+------------+------------------------+------------+---------------------------------------------------------------+
| 6     | limit             | ``-l``     | ``--limit``            | No         | Number of records to return, default -1 means return all data |
+-------+-------------------+------------+------------------------+------------+---------------------------------------------------------------+

-  *Examples*:

   .. code:: bash

       flow component output-data -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0 --output-path ./examples/

``output-data-table``
~~~~~~~~~~~~~~~~~~~~~

-  *Description*: View table name and namespace.
-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description      |
+=======+===================+============+========================+============+==================+
| 1     | job\_id           | ``-j``     | ``--job_id``           | Yes        | Job ID           |
+-------+-------------------+------------+------------------------+------------+------------------+
| 2     | role              | ``-r``     | ``--role``             | Yes        | Role             |
+-------+-------------------+------------+------------------------+------------+------------------+
| 3     | party\_id         | ``-p``     | ``--party_id``         | Yes        | Party ID         |
+-------+-------------------+------------+------------------------+------------+------------------+
| 4     | component\_name   | ``-cpn``   | ``--component_name``   | Yes        | Component Name   |
+-------+-------------------+------------+------------------------+------------+------------------+

-  *Examples*:

   .. code:: bash

       flow component output-data-table -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0

``list``
~~~~~~~~

-  *Description*: List components of a specified job.
-  *Arguments*:

+-------+------------+-----------+----------------+------------+---------------+
| No.   | Argument   | Flag\_1   | Flag\_2        | Required   | Description   |
+=======+============+===========+================+============+===============+
| 1     | job\_id    | ``-j``    | ``--job_id``   | Yes        | Job ID        |
+-------+------------+-----------+----------------+------------+---------------+

-  *Examples*:

.. code:: bash

    flow component list -j $JOB_ID

``get-summary``
~~~~~~~~~~~~~~~

-  *Description*: Download summary of a specified component and save it
   as a json file.
-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+----------------------------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description                            |
+=======+===================+============+========================+============+========================================+
| 1     | job\_id           | ``-j``     | ``--job_id``           | Yes        | Job ID                                 |
+-------+-------------------+------------+------------------------+------------+----------------------------------------+
| 2     | role              | ``-r``     | ``--role``             | Yes        | Role                                   |
+-------+-------------------+------------+------------------------+------------+----------------------------------------+
| 3     | party\_id         | ``-p``     | ``--party_id``         | Yes        | Party ID                               |
+-------+-------------------+------------+------------------------+------------+----------------------------------------+
| 4     | component\_name   | ``-cpn``   | ``--component_name``   | Yes        | Component Name                         |
+-------+-------------------+------------+------------------------+------------+----------------------------------------+
| 5     | output\_path      | ``-o``     | ``--output-path``      | No         | User specifies output directory path   |
+-------+-------------------+------------+------------------------+------------+----------------------------------------+

-  *Examples*:

.. code:: bash

    flow component get-summary -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0

    flow component get-summary -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0 -o ./examples/

Model
-----

``load``
~~~~~~~~

-  *Description*: Load model.
-  *Arguments*:

+-------+--------------+-----------+-------------------+------------+-----------------------------------+
| No.   | Argument     | Flag\_1   | Flag\_2           | Required   | Description                       |
+=======+==============+===========+===================+============+===================================+
| 1     | conf\_path   | ``-c``    | ``--conf-path``   | No         | Runtime configuration file path   |
+-------+--------------+-----------+-------------------+------------+-----------------------------------+
| 2     | job\_id      | ``-j``    | ``--job_id``      | No         | Job ID                            |
+-------+--------------+-----------+-------------------+------------+-----------------------------------+

-  *Examples*:

.. code:: bash

    flow model load -c fate_flow/examples/publish_load_model.json
    flow model load -j $JOB_ID

``bind``
~~~~~~~~

-  *Description*: Bind model.
-  *Arguments*:

+-------+--------------+-----------+-------------------+------------+-----------------------------------+
| No.   | Argument     | Flag\_1   | Flag\_2           | Required   | Description                       |
+=======+==============+===========+===================+============+===================================+
| 1     | conf\_path   | ``-c``    | ``--conf-path``   | Yes        | Runtime configuration file path   |
+-------+--------------+-----------+-------------------+------------+-----------------------------------+
| 2     | job\_id      | ``-j``    | ``--job_id``      | No         | Job ID                            |
+-------+--------------+-----------+-------------------+------------+-----------------------------------+

-  *Examples*:

.. code:: bash

    flow model bind -c fate_flow/examples/bind_model_service.json
    flow model bind -c fate_flow/examples/bind_model_service.json -j $JOB_ID

``import``
~~~~~~~~~~

-  *Description*: Import model
-  *Arguments*:

+-------+-----------------+-----------+-------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| No.   | Argument        | Flag\_1   | Flag\_2           | Required   | Description                                                                                                                                    |
+=======+=================+===========+===================+============+================================================================================================================================================+
| 1     | conf\_path      | ``-c``    | ``--conf-path``   | Yes        | Runtime configuration file path                                                                                                                |
+-------+-----------------+-----------+-------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| 2     | from-database   |           | --from-database   | No         | If specified and there is a valid database environment, fate flow will import model from database which you specified in configuration file.   |
+-------+-----------------+-----------+-------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------+

-  *Examples*:

.. code:: bash

    flow model import -c fate_flow/examples/import_model.json
    flow model import -c fate_flow/examples/restore_model.json --from-database

``export``
~~~~~~~~~~

-  *Description*: Export model
-  *Arguments*:

+-------+---------------+-----------+---------------------+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| No.   | Argument      | Flag\_1   | Flag\_2             | Required   | Description                                                                                                                                  |
+=======+===============+===========+=====================+============+==============================================================================================================================================+
| 1     | conf\_path    | ``-c``    | ``--conf-path``     | Yes        | Runtime configuration file path                                                                                                              |
+-------+---------------+-----------+---------------------+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| 2     | to-database   |           | ``--to-database``   | No         | If specified and there is a valid database environment, fate flow will export model to database which you specified in configuration file.   |
+-------+---------------+-----------+---------------------+------------+----------------------------------------------------------------------------------------------------------------------------------------------+

-  *Examples*:

.. code:: bash

    flow model export -c fate_flow/examples/export_model.json
    flow model export -c fate_flow/examplse/store_model.json --to-database

``migrate``
~~~~~~~~~~~

-  *Description*: Migrate model
-  *Arguments*:

+-------+--------------+-----------+-------------------+------------+-----------------------------------+
| No.   | Argument     | Flag\_1   | Flag\_2           | Required   | Description                       |
+=======+==============+===========+===================+============+===================================+
| 1     | conf\_path   | ``-c``    | ``--conf-path``   | Yes        | Runtime configuration file path   |
+-------+--------------+-----------+-------------------+------------+-----------------------------------+

-  *Examples*:

.. code:: bash

    flow model migrate -c fate_flow/examples/migrate_model.json

``tag-list``
~~~~~~~~~~~~

-  *Description*: List tags of model.
-  *Arguments*:

+-------+------------+-----------+----------------+------------+---------------+
| No.   | Argument   | Flag\_1   | Flag\_2        | Required   | Description   |
+=======+============+===========+================+============+===============+
| 1     | job\_id    | ``-j``    | ``--job_id``   | Yes        | Job ID        |
+-------+------------+-----------+----------------+------------+---------------+

-  *Examples*:

.. code:: bash

    flow model tag-list -j $JOB_ID

``tag-model``
~~~~~~~~~~~~~

-  *Description*: Tag model.
-  *Arguments*:

+-------+-------------+-----------+------------------+------------+--------------------------------------------------------------------------------------------------------+
| No.   | Argument    | Flag\_1   | Flag\_2          | Required   | Description                                                                                            |
+=======+=============+===========+==================+============+========================================================================================================+
| 1     | job\_id     | ``-j``    | ``--job_id``     | Yes        | Job ID                                                                                                 |
+-------+-------------+-----------+------------------+------------+--------------------------------------------------------------------------------------------------------+
| 2     | tag\_name   | ``-t``    | ``--tag-name``   | Yes        | The name of tag                                                                                        |
+-------+-------------+-----------+------------------+------------+--------------------------------------------------------------------------------------------------------+
| 3     | remove      |           | ``--remove``     | No         | If specified, the name of specified model will be removed from the model name list of specified tag.   |
+-------+-------------+-----------+------------------+------------+--------------------------------------------------------------------------------------------------------+

-  *Examples*:

.. code:: bash

    flow model tag-model -j $JOB_ID -t $TAG_NAME
    flow model tag-model -j $JOB_ID -t $TAG_NAME --remove

``deploy``
~~~~~~~~~~~

-  *Description*: Deploy model.
-  *Arguments*:

+-------+--------------------+-----------+------------------------+------------+----------------------------------------------------------------+
| No.   | Argument           | Flag\_1   | Flag\_2                | Required   | Description                                                    |
+=======+====================+===========+========================+============+================================================================+
| 1     | model\_id          |           | ``--model-id``         | Yes        | Parent model id.                                               |
+-------+--------------------+-----------+------------------------+------------+----------------------------------------------------------------+
| 2     | model\_version     |           | ``--model-version``    | Yes        | Parent model version.                                          |
+-------+--------------------+-----------+------------------------+------------+----------------------------------------------------------------+
| 3     | cpn\_list          |           | ``--cpn-list``         | No         | User inputs a string to specify component list.                |
+-------+--------------------+-----------+------------------------+------------+----------------------------------------------------------------+
| 4     | cpn\_path          |           | ``--cpn-path``         | No         | User specifies a file path which records the component list.   |
+-------+--------------------+-----------+------------------------+------------+----------------------------------------------------------------+
| 5     | dsl\_path          |           | ``--train-dsl-path``   | No         | User specified predict dsl file.                               |
+-------+--------------------+-----------+------------------------+------------+----------------------------------------------------------------+

-  *Examples*:

.. code:: bash

    flow model deploy --model-id $MODEL_ID --model-version $MODEL_VERSION

``get-predict-dsl``
~~~~~~~~~~~~~~~~~~~~

-  *Description*: Get predict dsl of model.
-  *Arguments*:

+-------+--------------------+-----------+--------------------+------------+--------------------------+
| No.   | Argument           | Flag\_1   | Flag\_2            | Required   | Description              |
+=======+====================+===========+====================+============+==========================+
| 1     | model\_id          |           | ``--model-id``     | Yes        | Model id                 |
+-------+--------------------+-----------+--------------------+------------+--------------------------+
| 2     | model\_version     |           | ``--model-version``| Yes        | Model version            |
+-------+--------------------+-----------+--------------------+------------+--------------------------+
| 3     | output\_path       | ``-o``    | ``--output-path``  | Yes        | Output directory path    |
+-------+--------------------+-----------+--------------------+------------+--------------------------+

-  *Examples*:

.. code:: bash

    flow model get-predict-dsl --model-id $MODEL_ID --model-version $MODEL_VERSION -o ./examples/

``get-predict-conf``
~~~~~~~~~~~~~~~~~~~~

-  *Description*: Get predict conf template of model.
-  *Arguments*:

+-------+--------------------+-----------+--------------------+------------+--------------------------+
| No.   | Argument           | Flag\_1   | Flag\_2            | Required   | Description              |
+=======+====================+===========+====================+============+==========================+
| 1     | model\_id          |           | ``--model-id``     | Yes        | Model id                 |
+-------+--------------------+-----------+--------------------+------------+--------------------------+
| 2     | model\_version     |           | ``--model-version``| Yes        | Model version            |
+-------+--------------------+-----------+--------------------+------------+--------------------------+
| 3     | output\_path       | ``-o``    | ``--output-path``  | Yes        | Output directory path    |
+-------+--------------------+-----------+--------------------+------------+--------------------------+

-  *Examples*:

.. code:: bash

    flow model get-predict-conf --model-id $MODEL_ID --model-version $MODEL_VERSION -o ./examples/


``get-model-info``
~~~~~~~~~~~~~~~~~~~~

-  *Description*: Get information of model.
-  *Arguments*:

+-------+--------------------+-----------+--------------------+------------+--------------------------+
| No.   | Argument           | Flag\_1   | Flag\_2            | Required   | Description              |
+=======+====================+===========+====================+============+==========================+
| 1     | model\_id          |           | ``--model-id``     | No         | Model id                 |
+-------+--------------------+-----------+--------------------+------------+--------------------------+
| 2     | model\_version     |           | ``--model-version``| Yes        | Model version            |
+-------+--------------------+-----------+--------------------+------------+--------------------------+
| 3     | role               | ``-r``    | ``--role``         | No         | Role                     |
+-------+--------------------+-----------+--------------------+------------+--------------------------+
| 2     | party\_id          | ``-p``    | ``--party-id``     | No         | Party ID                 |
+-------+--------------------+-----------+--------------------+------------+--------------------------+
| 3     | detail             |           | ``--detail``       | No         | Show details             |
+-------+--------------------+-----------+--------------------+------------+--------------------------+

-  *Examples*:

.. code:: bash

    flow model get-model-info --model-id $MODEL_ID --model-version $MODEL_VERSION
    flow model get-model-info --model-id $MODEL_ID --model-version $MODEL_VERSION --detail


Tag
---

``create``
~~~~~~~~~~

-  *Description*: Create tag.
-  *Arguments*:

+-------+--------------------+-----------+------------------+------------+--------------------------+
| No.   | Argument           | Flag\_1   | Flag\_2          | Required   | Description              |
+=======+====================+===========+==================+============+==========================+
| 1     | tag\_name          | ``-t``    | ``--tag-name``   | Yes        | The name of tag          |
+-------+--------------------+-----------+------------------+------------+--------------------------+
| 2     | tag\_description   | ``-d``    | ``--tag-desc``   | No         | The description of tag   |
+-------+--------------------+-----------+------------------+------------+--------------------------+

-  *Examples*:

.. code:: bash

    flow tag create -t tag1 -d "This is the description of tag1."
    flow tag create -t tag2

``update``
~~~~~~~~~~

-  *Description*: Update information of tag.
-  *Arguments*:

+-------+-------------------------+-----------+----------------------+------------+--------------------------+
| No.   | Argument                | Flag\_1   | Flag\_2              | Required   | Description              |
+=======+=========================+===========+======================+============+==========================+
| 1     | tag\_name               | ``-t``    | ``--tag-name``       | Yes        | The name of tag          |
+-------+-------------------------+-----------+----------------------+------------+--------------------------+
| 2     | new\_tag\_name          |           | ``--new-tag-name``   | No         | New name of tag          |
+-------+-------------------------+-----------+----------------------+------------+--------------------------+
| 3     | new\_tag\_description   |           | ``--new-tag-desc``   | No         | New description of tag   |
+-------+-------------------------+-----------+----------------------+------------+--------------------------+

-  *Examples*:

.. code:: bash

    flow tag update -t tag1 --new-tag-name tag2
    flow tag update -t tag1 --new-tag-desc "This is the new description."

``list``
~~~~~~~~

-  *Description*: List recorded tags.
-  *Arguments*:

+-------+------------+-----------+---------------+------------+----------------------------------------------+
| No.   | Argument   | Flag\_1   | Flag\_2       | Required   | Description                                  |
+=======+============+===========+===============+============+==============================================+
| 1     | limit      | ``-l``    | ``--limit``   | No         | Number of records to return. (default: 10)   |
+-------+------------+-----------+---------------+------------+----------------------------------------------+

-  *Examples*:

.. code:: bash

    flow tag list
    flow tag list -l 3

``query``
~~~~~~~~~

-  *Description*: Retrieve tag.
-  *Arguments*:

+-------+---------------+-----------+--------------------+------------+------------------------------------------------------------------------------------------------+
| No.   | Argument      | Flag\_1   | Flag\_2            | Required   | Description                                                                                    |
+=======+===============+===========+====================+============+================================================================================================+
| 1     | tag\_name     | ``-t``    | ``--tag-name``     | Yes        | The name of tag                                                                                |
+-------+---------------+-----------+--------------------+------------+------------------------------------------------------------------------------------------------+
| 2     | with\_model   |           | ``--with-model``   | No         | If specified, the information of models which have the tag custom queried would be displayed   |
+-------+---------------+-----------+--------------------+------------+------------------------------------------------------------------------------------------------+

-  *Examples*:

.. code:: bash

    flow tag query -t $TAG_NAME
    flow tag query -t $TAG_NAME --with-model

``delete``
~~~~~~~~~~

-  *Description*: Delete tag.
-  *Arguments*:

+-------+-------------+-----------+------------------+------------+-------------------+
| No.   | Argument    | Flag\_1   | Flag\_2          | Required   | Description       |
+=======+=============+===========+==================+============+===================+
| 1     | tag\_name   | ``-t``    | ``--tag-name``   | Yes        | The name of tag   |
+-------+-------------+-----------+------------------+------------+-------------------+

-  *Examples*:

.. code:: bash

    flow tag delete -t tag1

Data
----

``download``
~~~~~~~~~~~~

-  *Description*: Download Data Table.
-  *Arguments*:

+-------+--------------+-----------+-------------------+------------+---------------------------+
| No.   | Argument     | Flag\_1   | Flag\_2           | Required   | Description               |
+=======+==============+===========+===================+============+===========================+
| 1     | conf\_path   | ``-c``    | ``--conf-path``   | Yes        | Configuration file path   |
+-------+--------------+-----------+-------------------+------------+---------------------------+

-  *Examples*:

.. code:: bash

    flow data download -c fate_flow/examples/download_host.json

``upload``
~~~~~~~~~~

-  *Description*: Upload Data Table.
-  *Arguments*:

+-------+--------------+-----------+-------------------+------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
| No.   | Argument     | Flag\_1   | Flag\_2           | Required   | Description                                                                                                                                      |
+=======+==============+===========+===================+============+==================================================================================================================================================+
| 1     | conf\_path   | ``-c``    | ``--conf-path``   | Yes        | Configuration file path                                                                                                                          |
+-------+--------------+-----------+-------------------+------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
| 2     | verbose      |           | ``--verbose``     | No         | If specified, verbose mode will be turn on. Users can have feedback on upload task in progress. (Default: False)                                 |
+-------+--------------+-----------+-------------------+------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
| 3     | drop         |           | ``--drop``        | No         | If specified, data of old version would be replaced by the current version. Otherwise, current upload task would be rejected. (Default: False)   |
+-------+--------------+-----------+-------------------+------------+--------------------------------------------------------------------------------------------------------------------------------------------------+

-  *Examples*:

.. code:: bash

    flow data upload -c fate_flow/examples/upload_guest.json
    flow data upload -c fate_flow/examples/upload_host.json --verbose --drop

``upload-history``
~~~~~~~~~~~~~~~~~~

-  *Description*: Query Upload Table History.
-  *Arguments*:

+-------+------------+-----------+----------------+------------+----------------------------------------------+
| No.   | Argument   | Flag\_1   | Flag\_2        | Required   | Description                                  |
+=======+============+===========+================+============+==============================================+
| 1     | limit      | ``-l``    | ``--limit``    | No         | Number of records to return. (default: 10)   |
+-------+------------+-----------+----------------+------------+----------------------------------------------+
| 2     | job\_id    | ``-j``    | ``--job_id``   | No         | Job ID                                       |
+-------+------------+-----------+----------------+------------+----------------------------------------------+

-  *Examples*:

.. code:: bash

    flow data upload-history -l 20
    flow data upload-history --job-id $JOB_ID

Task
----

``query``
~~~~~~~~~

-  *Description*: Query task information by filters.
-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description      |
+=======+===================+============+========================+============+==================+
| 1     | job\_id           | ``-j``     | ``--job_id``           | No         | Job ID           |
+-------+-------------------+------------+------------------------+------------+------------------+
| 2     | role              | ``-r``     | ``--role``             | No         | Role             |
+-------+-------------------+------------+------------------------+------------+------------------+
| 3     | party\_id         | ``-p``     | ``--party_id``         | No         | Party ID         |
+-------+-------------------+------------+------------------------+------------+------------------+
| 4     | component\_name   | ``-cpn``   | ``--component_name``   | No         | Component Name   |
+-------+-------------------+------------+------------------------+------------+------------------+
| 5     | status            | ``-s``     | ``--status``           | No         | Job Status       |
+-------+-------------------+------------+------------------------+------------+------------------+

-  *Examples*:

.. code:: bash

    flow task query -j $JOB_ID -p 9999 -r guest
    flow task query -cpn hetero_feature_binning_0 -s complete

``list``
~~~~~~~~

-  *Description*: List tasks.
-  *Arguments*:

+-------+------------+-----------+---------------+------------+----------------------------------------------+
| No.   | Argument   | Flag\_1   | Flag\_2       | Required   | Description                                  |
+=======+============+===========+===============+============+==============================================+
| 1     | limit      | ``-l``    | ``--limit``   | No         | Number of records to return. (default: 10)   |
+-------+------------+-----------+---------------+------------+----------------------------------------------+

-  *Examples*:

.. code:: bash

    flow task list
    flow task list -l 25

Table
-----

``info``
~~~~~~~~

-  *Description*: Query Table Information.
-  *Arguments*:

+-------+---------------+-----------+--------------------+------------+---------------+
| No.   | Argument      | Flag\_1   | Flag\_2            | Required   | Description   |
+=======+===============+===========+====================+============+===============+
| 1     | namespace     | ``-n``    | ``--namespace``    | Yes        | Namespace     |
+-------+---------------+-----------+--------------------+------------+---------------+
| 2     | table\_name   | ``-t``    | ``--table-name``   | Yes        | Table Name    |
+-------+---------------+-----------+--------------------+------------+---------------+

-  *Examples*:

.. code:: bash

    flow table info -n $NAMESPACE -t $TABLE_NAME

``delete``
~~~~~~~~~~

-  *Description*: Delete A Specified Table.
-  *Arguments*:

+-------+-------------------+------------+------------------------+------------+------------------+
| No.   | Argument          | Flag\_1    | Flag\_2                | Required   | Description      |
+=======+===================+============+========================+============+==================+
| 1     | namespace         | ``-n``     | ``--namespace``        | No         | Namespace        |
+-------+-------------------+------------+------------------------+------------+------------------+
| 2     | table\_name       | ``-t``     | ``--table_name``       | No         | Table name       |
+-------+-------------------+------------+------------------------+------------+------------------+


-  *Examples*:

.. code:: bash

    flow table delete -n $NAMESPACE -t $TABLE_NAME


Queue
-----

``clean``
~~~~~~~~~

-  *Description*: Cancel all jobs in queue.
-  *Arguments*: None.
-  *Examples*:

.. code:: bash

    flow queue clean

