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

/v1/job/log
~~~~~~~~~~~

-  request structure

   -  job_id: Required, String: job id

-  response structure

   -  data: Object

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
   -  component_name: Required,String: conponent name

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
   -  meric_name: Required,String: meric name
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

-  response structure

   -  retcode: return code,Integer
   -  retmsg: return code description,String
   -  data: pipeline dag dependency data,Object

Model
~~~~~

/v1/model/load
~~~~~~~~~~~~~~

-  request structure

   -  initiator: Required,Object: job initiator information
   -  role: Required,Object: role information
   -  model: Requied,Object: model information

-  response structure

   -  job_id:job id, String
   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: status info, Object

/v1/model/bind
~~~~~~~~~~~~~~

-  request structure

   -  service_id: Required,String: service id
   -  initiator: Required,Object: job initiator information
   -  job_parameters: Required ,Object: model id and model version
   -  role: Required,Object: role information
   -  servings: Optional,Array: my party servings

-  response structure

   -  retcode: return code, Integer

/v1/model/version
~~~~~~~~~~~~~~~~~

-  request structure

   -  namespace: Requied,String: data table namespace

-  response structure

   -  data: version history,Array

/v1/model/transfer
~~~~~~~~~~~~~~~~~~

-  request structure

   -  name: Requied,String: data table name
   -  namespace: Requied,String: data table namespace

-  response structure

   -  retcode: return code, Integer
   -  retmsg: return code description, String
   -  data: model data, Object

Table
-----

/v1/table/table_info
~~~~~~~~~~~~~~~~~~~~

-  request structure

   -  create: Optional, Boolean: whether to create
   -  namespace: Optional,String: download data table namespace
   -  table_name: Optional,String: download data table name
   -  local: Required,Object: local configuration
   -  role: Optional,Object: role information
   -  data_type: Optional,String: download file data type
   -  gen_table_info: Optional,Boolean: tag table information

-  response structure

   -  data: table information
