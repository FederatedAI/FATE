FATE-Flow REST API
==================

-  HTTP Method: POST
-  Content-Type: application/json

DataAccess
----------

/v1/data/upload
~~~~~~~~~~~~~~~

-  request structure

   -  namespace: Required,String: upload data table namespace
   -  table_name: Required,String: upload data table name
   -  work_mode: Required,Integer: eggroll’s working mode
   -  file: Required, String: upload file location
   -  head: Required,Integer: determine if there is a data header
   -  partition: Required,Integer: set the number of partitions to save
      data
   -  module: Optional,String: If you need to use the data of the
      machine where the FATE-Flow server is located, this value is not
      empty.
   -  use_local_data: Optional,String: If you need to use the data of the machine where the FATE-Flow server is located, this value is 0.
   -  drop: Optional, Integer: When the cluster deployment uses the same table to upload data, it is necessary to carry the drop parameter,0 represents overwriting upload, 1 represents deleting the previous data and re-uploading


-  response structure

   -  job_id: upload job id,String
   -  data: return data for submitting job ,Object

/v1/data/download
~~~~~~~~~~~~~~~~~

-  request structure

   -  namespace: Required,String: download data table namespace
   -  table_name: Required,String: download data table name
   -  output_path: Required, String: download file location
   -  work_mode: Required,Integer:working mode
   -  delimitor: Optional,String: download data delimitor

-  response structure

   -  job_id: download job id,String
   -  data: return data for submitting job ,Object

/v1/data/upload/history
~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   -  job_id: Optional,String:download job id
   -  limit: Optional, Integer:Limit quantity

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: return data for submitting job ,Object

Job
---

/v1/job/submit
~~~~~~~~~~~~~~

-  request structure

   -  job_runtime_conf: Required,Object: configuration information for
      the currently submitted job
   -  job_dsl: Required,Object: dsl of the currently submitted job

-  response structure

   -  job_id: job id of the currently submitted job,String
   -  data: return data for submitting job ,Object

/v1/job/stop
~~~~~~~~~~~~

-  request structure

   -  job_id: Required, String: job id

-  response structure

   -  job_id: job id of the currently submitted job,String
   -  retmsg: return code description,String

/v1/job/query
~~~~~~~~~~~~~

-  request structure

   -  job_id: Optional,String: job id
   -  name: Optional,String: job name
   -  description: Optional,String: job description
   -  tag: Optional,String:Optional,String: job tag
   -  role: Optional,String: job role
   -  party_id: Optional,String: job party id
   -  roles: Optional,String: job roles
   -  initiator_party_id: Optional,String: initiator’s party id
   -  is_initiator: Optional,Integer: mark if it is the initiator
   -  dsl: Optional,String: job dsl
   -  runtime_conf : Optional,String: configuration information for the
      job
   -  run_ip: Optional,String: job run ip
   -  status: Optional,String: job status
   -  current_steps: Optional,String:record component id in DSL
   -  current_tasks: Optional,String: record task id
   -  progress: Optional,Integer: job progress
   -  create_time: Optional,Integer: job create time
   -  update_time: Optional,Integer:job update time
   -  start_time: Optional,Integer: job start time
   -  end_time: Optional,Integer: job end time
   -  elapsed: Optional,Integer: job elapsed time

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: job data, Array

/v1/job/update
~~~~~~~~~~~~~~

-  request structure

   -  job_id: Required,String: job id
   -  role: Required,String: job role
   -  party_id: Required,String: job party id
   -  notes: Required, String: remark Information

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String

/v1/job/config
~~~~~~~~~~~~~~

-  request structure

   -  job_id: Optional,String: job id
   -  name: Optional,String: job name
   -  description: Optional,String: job description
   -  tag: Optional,String:Optional,String: job tag
   -  role: Optional,String: job role
   -  party_id: Optional,String: job party id
   -  roles: Optional,String: job roles
   -  initiator_party_id: Optional,String: initiator’s party id
   -  is_initiator: Optional,Integer: mark if it is the initiator
   -  dsl: Optional,String: job dsl
   -  runtime_conf : Optional,String: configuration information for the
      job
   -  run_ip: Optional,String: job run ip
   -  status: Optional,String: job status
   -  current_steps: Optional,String:record component id in DSL
   -  current_tasks: Optional,String: record task id
   -  progress: Optional,Integer: job progress
   -  create_time: Optional,Integer: job create time
   -  update_time: Optional,Integer:job update time
   -  start_time: Optional,Integer: job start time
   -  end_time: Optional,Integer: job end time
   -  elapsed: Optional,Integer: job elapsed time

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: config data, Object


/v1/job/task/query
~~~~~~~~~~~~~~~~~~

-  request structure

   -  job_id: Optional,String: job id
   -  name: Optional,String: job name
   -  description: Optional,String: job description
   -  tag: Optional,String:Optional,String: job tag
   -  role: Optional,String: job role
   -  party_id: Optional,String: job party id
   -  roles: Optional,String: job roles
   -  initiator_party_id: Optional,String: initiator’s party id
   -  is_initiator: Optional,Integer: mark if it is the initiator
   -  dsl: Optional,String: job dsl
   -  runtime_conf : Optional,String: configuration information for the
      job
   -  run_ip: Optional,String: job run ip
   -  status: Optional,String: job status
   -  current_steps: Optional,String:record component id in DSL
   -  current_tasks: Optional,String: record task id
   -  progress: Optional,Integer: job progress
   -  create_time: Optional,Integer: job create time
   -  update_time: Optional,Integer:job update time
   -  start_time: Optional,Integer: job start time
   -  end_time: Optional,Integer: job end time
   -  elapsed: Optional,Integer: job elapsed time

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: tasks data, Array


/v1/job/list/job
~~~~~~~~~~~~~~~~~~

-  request structure

   - limit: Optional, Integer: limitation of number of return records

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: info of jobs, Array


/v1/job/list/task
~~~~~~~~~~~~~~~~~~

-  request structure

   - limit: Optional, Integer: limitation of number of return records

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: info of tasks, Array


/v1/job/dsl/generate
~~~~~~~~~~~~~~~~~~~~~

-  request structure

   - train_dsl: Required, String: training dsl
   - cpn_str: Required, String or Array: list of fate_components which are chose to be used
   - filename: Optional, String: generated dsl storing path

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: generated dsl, Array



Tracking
~~~~~~~~

/v1/tracking/job/data_view
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   -  job_id: Required,String: job id
   -  role: Required,String: role information
   -  party_id: Required,Integer: party id

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: job view data,Object

/v1/tracking/component/metric/all
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   -  job_id: Required,String: job id
   -  role: Required,String: role information
   -  party_id: Required,Integer
   -  component_name: Required,String: conponent name

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: all metric data,Object

/v1/tracking/component/metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   -  job_id: Required,String: job id
   -  role: Required,String: role information
   -  party_id: Required,Integer
   -  component_name: Required,String: component name

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: metrics data,Object

/v1/tracking/component/metric_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   -  job_id: Required,String: job id
   -  role: Required,String: role information
   -  party_id: Required,Integer: party id
   -  component_name: Required,String: component name
   -  meric_name: Required,String: metric name
   -  metric_namespace: Required,String: metric namespace

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: metric data, Array
   -  meta: metric meta, Object

/v1/tracking/component/parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   -  job_id: Required,String: job id
   -  role: Required,String: role information
   -  party_id: Required,Integer: party id
   -  component_name: Required,String: component name

-  response structure

   -  retcode:return code,Integer
   -  retmsg: return code description,String
   -  data: output parameters, Object

/v1/tracking/component/output/model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   -  job_id: Required,String: job id
   -  role: Required,String: role information
   -  party_id: Required,Integer: party id
   -  component_name: Required,String: component name

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: output model, Object
   -  meta: component model meta,Object

/v1/tracking/component/output/data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   -  job_id: Required,String: job id
   -  role: Required,String: role information
   -  party_id: Required,Integer: party id
   -  component_name: Required,String: component name

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: output data, Array
   -  meta: schema header information, Object

Pipeline
~~~~~~~~

/v1/pipeline/dag/dependency
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   -  job_id: Required,String:job id
   -  role: Required,String: role information
   -  party_id: Required,Integer: party id

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: pipeline dag dependency data,Object

Model
~~~~~

/v1/model/load
~~~~~~~~~~~~~~

-  request structure

   -  initiator: Required,Object: job initiator information, including party_id and role
   -  job_parameters: Required,Object: job parameters information, including work_mode, model_id and model_version
   -  role: Required,Object: role information of the parties
   -  servings: Optional,Array: fate serving address and port

-  response structure

   -  job_id:job id, String
   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: status info, Object

/v1/model/bind
~~~~~~~~~~~~~~

-  request structure

   -  service_id: Required,String: service id
   -  initiator: Required,Object: job initiator information, including party_id and role
   -  job_parameters: Required,Object: job parameters information, including work_mode, model_id and model_version
   -  role: Required,Object: role information of the parties
   -  servings: Optional,Array: fate serving address and port

-  response structure

   -  retcode: return code, Integer


/v1/model/transfer
~~~~~~~~~~~~~~~~~~

-  request structure

   -  name: Requied,String: model version
   -  namespace: Requied,String: model id

-  response structure

   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: model data, Object


/v1/model/import
~~~~~~~~~~~~~~~~~

-  request structure

   - model_version: Required, Integer: model version
   - model_id: Required, String: model id
   - role: Required, String: role
   - party_id: Required, String: party id

-  response structure

   - retcode: return code, Integer
   - retmsg: return code description, String


/v1/model/export
~~~~~~~~~~~~~~~~~

-  request structure

   - model_version: Required, Integer: model version
   - model_id: Required, String: model id
   - role: Required, String: role
   - party_id: Required, String: party id

-  response structure

   - retcode: return code, Integer
   - retmsg: return code description, String


/v1/model/store
~~~~~~~~~~~~~~~~~

-  request structure

   - model_version: Required, Integer: model version
   - model_id: Required, String: model id
   - role: Required, String: role
   - party_id: Required, String: party id

-  response structure

   - retcode: return code, Integer
   - retmsg: return code description, String


/v1/model/restore
~~~~~~~~~~~~~~~~~

-  request structure

   - model_version: Required, Integer: model version
   - model_id: Required, String: model id
   - role: Required, String: role
   - party_id: Required, String: party id

-  response structure

   - retcode: return code, Integer
   - retmsg: return code description, String


/v1/model/model_tag/retrieve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   - job_id: Required, Integer: a valid job id or model version

-  response structure

   - retcode: return code, Integer
   - retmsg: return code description, String
   - data: information of tags related to the specified model


/v1/model/model_tag/create
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   - job_id: Required, Integer: a valid job id or model version
   - tag_name: Required, String: a valid name of tag

-  response structure

   - retcode: return code, Integer
   - retmsg: return code description, String


/v1/model/model_tag/remove
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   - job_id: Required, Integer: a valid job id or model version
   - tag_name: Required, String: a valid name of tag

-  response structure

   - retcode: return code, Integer
   - retmsg: return code description, String


/v1/model/tag/retrieve
~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   - tag_name: Required, String: a valid tag name
   - with_model: Optional, Boolean: choose to show tag info or tag info related to models

-  response structure

   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: tag info, Object


/v1/model/tag/create
~~~~~~~~~~~~~~~~~~~~~

-  request structure

   - tag_name: Required, String: name of tag
   - tag_desc: Optional, String: description of tag

-  response structure

   - retcode: return code, Integer
   - retmsg: return code description, String


/v1/model/tag/destroy
~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   - tag_name: Required, String: a valid tag name

-  response structure

   - retcode: return code, Integer
   - retmsg: return code description, String


/v1/model/tag/update
~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   - tag_name: Required, String: a valid tag name
   - new_tag_name: Optional, String: a new name to replace previous name
   - new_tag_desc: Optional, String: a new decription to replace previous description

-  response structure

   - retcode: return code, Integer
   - retmsg: return code description, String


/v1/model/tag/list
~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   - limit: Required, Integer: limitation of number of return records

-  response structure

   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: tag info, Object


/v1/model/migrate
~~~~~~~~~~~~~~~~~~

-  request structure

   - migrate_initiator: Required, Object: indicates which party is the new initiator after migrating
   - unify_model_version: Optional, String: a unitive model version for migrate model
   - role: Required, String: information of roles which participated in model training, including role name and array of party ids
   - migrate_role: Required, Object: information of roles model would be migrated to, including role name and array of party ids
   - model_id: Required, String: original model id
   - model_version: Required, Integer: original model version
   - execute_party: Required, Object: parties that is going to execute model migration task
   - job_parameters: Required, Object: job parameters information, including work_mode, model_id and model_version

-  response structure

   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: status info, Object


/v1/model/query
~~~~~~~~~~~~~~~~~~

-  request structure

   - model_version: Required, Integer: model version
   - model_id: Optional, String: model id
   - role: Optional, String: role
   - party_id: Optional, String: party id
   - query_filters: Optional, Array: features filters

-  response structure

   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: model info, Object


/v1/model/deploy
~~~~~~~~~~~~~~~~~~

-  request structure

   - model_version: Required, Integer: model version
   - model_id: Required, String: model id
   - cpn_list: Optional, String: array-like string that contains fate_components
   - cpn_path: Optional, String: file path of plain text which stores component list
   - dsl_path: Optional, String: file path of plain text which stores dsl content

-  response structure

   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: status info, Object


/v1/model/get/predict/dsl
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   - model_version: Required, Integer: model version
   - model_id: Optional, String: model id
   - role: Optional, String: role
   - party_id: Optional, String: party id

-  response structure

   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: predict dsl of specified model, Object


/v1/model/get/predict/conf
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   - model_version: Required, Integer: model version
   - model_id: Required, String: model id
   - filename: Optional, String: file storing path

-  response structure

   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: predict config of specified model, Object


/v1/model/homo/convert
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  request structure

   - model_version: Required, Integer: model version
   - model_id: Required, String: model id
   - role: Required, String: role
   - party_id: Required, String: party id
   - framework_name: Optional, String: the target machine learning framework this model should be converted to, will be inferred automatically if it is empty

-  response structure

   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: converted component name and the saved file path, List


/v1/model/homo/deploy
~~~~~~~~~~~~~~~~~~

-  request structure

   - service_id: Required, String: service id
   - model_version: Required, Integer: model version
   - model_id: Required, String: model id
   - role: Required, String: role
   - party_id: Required, String: party id
   - component_name: Required, String: component name
   - framework_name: Optional, String: the target machine learning framework this serving service is for, will be inferred automatically if it is empty
   - deployment_type: Required, String: type of the serving service, only "kfserving" is allowed currently
   - deployment_parameters: Required, Object: parameters that will be configured for the serving service

-  response structure

   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: detailed info of the deployed serving service, Object



Table
-----

/v1/table/table_info
~~~~~~~~~~~~~~~~~~~~

-  request structure

   -  create: Optional, Boolean: whether to create
   -  namespace: Optional,String: download data table namespace, need to be used with table_name
   -  table_name: Optional,String: download data table name, need to be used with namespace
   -  local: Optional,Object: local configuration
   -  role: Optional,Object: role information
   -  data_type: Optional,String: download file data type
   -  gen_table_info: Optional,Boolean: tag table information

-  response structure

   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: table information

/v1/table/delete
~~~~~~~~~~~~~~~~~~~~

-  request structure

   - namespace: Optional,String: download data table namespace, need to be used with table_name
   - table_name:  Optional,String: download data table name, need to be used with namespace

-  response structure

   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: table information
