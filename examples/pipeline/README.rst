Pipeline Examples
=================

Introduction
-------------

We provide some example scripts of running
FATE jobs with `FATE-Pipeline <../../python/fate_client/README.rst>`_.

Please refer to the document linked above for details on FATE-Pipeline and FATE-Flow CLI v2.
DSL version of provided Pipeline examples can be found `here <../dsl/v2>`_.


Quick Start
-----------

Here is a general guide to quick start a FATE job. In this guide, default installation location of
FATE is `/data/projects/fate`.

1. (optional) create virtual env

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate
      pip install -U pip


2. install fate_client

   .. code-block:: bash

      # this step installs FATE-Pipeline, FATE-Flow CLI v2, and FATE-Flow SDK
      pip install fate_client
      pipeline init --help


3. provide server ip/port information of deployed FATE-Flow

   .. code-block:: bash

      # provide real ip address and port info of fate-flow server to initialize pipeline. Typically, the default ip and port are 127.0.0.1:8080.
      pipeline init --ip 127.0.0.1 --port 9380
      # optionally, set log directory of Pipeline
      cd /data/projects/fate/python/fate_client/pipeline
      pipeline init --ip 127.0.0.1 --port 9380 --log-directory ./logs

4. upload data with FATE-Pipeline

Before start a modeling task, the data to be used should be uploaded. Typically, a party is usually a cluster which include multiple nodes. Thus, when we upload these data, the data will be allocated to those nodes.

We have provided an example script to upload data:  `here <./demo/pipeline-upload.py>`_.

Set your fate path in the beginning of this script:

   .. code-block:: python
        #  path to data
        #  default fate installation path
        DATA_BASE = "/data/projects/fate"

User may modify file path and table name to upload arbitrary data following instructions in the script.

   .. code-block:: python

            pipeline_upload.add_upload_data(file=os.path.join(data_base, "examples/data/breast_hetero_guest.csv"),
                                    table_name=dense_data["name"],             # table name
                                    namespace=dense_data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info
            pipeline_upload.add_upload_data(file=os.path.join(data_base, "examples/data/breast_hetero_host.csv"),
                                    table_name=tag_data["name"],
                                    namespace=tag_data["namespace"],
                                    head=1, partition=partition)


For a list of available example data and general guide on table naming, please refer
   to this `guide <../data/README.md>`_.

   .. code-block:: bash

      #  upload demo data to FATE data storage, optionally provide directory where deployed examples/data locates
      cd /data/projects/fate
      python examples/pipeline/demo/pipeline-upload.py --base /data/projects/fate

   If upload job is invoked correctly, job id will be printed to terminal and an upload bar is shown.
   If FATE-Board is available, job progress can be monitored on Board as well.

   ::

         UPLOADING:||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||100.00%
         2021-03-25 17:13:21.548 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:121 - Job id is 202103251713214312523
                            Job is still waiting, time elapse: 0:00:01
         2021-03-25 17:13:23Running component upload_0, time elapse: 0:00:03
         2021-03-25 17:13:25.168 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:129 - Job is success!!! Job id is 202103251713214312523
         2021-03-25 17:13:25.169 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:130 - Total time: 0:00:03
         UPLOADING:||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||100.00%
         2021-03-25 17:13:25.348 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:121 - Job id is 202103251713251765644
                            Job is still waiting, time elapse: 0:00:01
         2021-03-25 17:13:27Running component upload_0, time elapse: 0:00:03
         2021-03-25 17:13:29.480 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:129 - Job is success!!! Job id is 202103251713251765644
         2021-03-25 17:13:29.480 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:130 - Total time: 0:00:04

    If you would like to change this demo and use your own data, please

5. run a FATE-Pipeline fit job

   .. code-block:: bash

      cd /data/projects/fate
      python examples/pipeline/demo/pipeline-quick-demo.py

The details of each step of this demo can be shown `here <./demo/pipeline-quick-demo.py>`_.

   This quick demo shows how to build to a heterogeneous SecureBoost job using uploaded data from previous step.
   Note that data are uploaded to the same machine in the previous step. To run the below job with cluster deployment,
   make sure to first upload data to corresponding parties and set role information and job parameters accordingly.

Progress of job execution will be printed as modules run.
   A message indicating final status ("success") will be printed when job finishes.
   The script queries final model information when model training completes.

   ::

        2021-03-25 17:13:51.370 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:121 - Job id is 202103251713510969875
                            Job is still waiting, time elapse: 0:00:00
        2021-03-25 17:13:52Running component reader_0, time elapse: 0:00:02
        2021-03-25 17:13:54Running component dataio_0, time elapse: 0:00:05
        2021-03-25 17:13:57Running component intersection_0, time elapse: 0:00:09
        2021-03-25 17:14:01Running component hetero_secureboost_0, time elapse: 0:00:35
        2021-03-25 17:14:27Running component evaluation_0, time elapse: 0:00:40
        2021-03-25 17:14:32.446 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:129 - Job is success!!! Job id is 202103251713510969875
        2021-03-25 17:14:32.447 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:130 - Total time: 0:00:41

6. (another example) run FATE-Pipeline fit and predict jobs

   .. code-block:: bash

      cd /data/projects/fate
      python examples/pipeline/demo/pipeline-mini-demo.py

   This `script <./demo/pipeline-mini-demo.py>`_ trains a heterogeneous logistic regression model and then runs prediction with the trained model.

   ::

        2021-03-25 17:16:24.832 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:121 - Job id is 202103251716244738746
                            Job is still waiting, time elapse: 0:00:00
        2021-03-25 17:16:25Running component reader_0, time elapse: 0:00:02
        2021-03-25 17:16:27Running component dataio_0, time elapse: 0:00:05
        2021-03-25 17:16:30Running component intersection_0, time elapse: 0:00:09
        2021-03-25 17:16:35Running component hetero_lr_0, time elapse: 0:00:38
        2021-03-25 17:17:04.332 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:129 - Job is success!!! Job id is 202103251716244738746
        2021-03-25 17:17:04.332 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:130 - Total time: 0:00:39

   Once fit job completes, demo script will print coefficients and training information of model.

   After having completed the fit job, script will invoke a predict job with the trained model.
   Note that ``Evaluation`` component is added to the prediction workflow. For more information on using
   FATE-Pipeline, please refer to this `guide <../../python/fate_client/pipeline/README.rst>`_.

   ::

        2021-03-25 17:17:05.568 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:121 - Job id is 202103251717052325809
                            Job is still waiting, time elapse: 0:00:01
        2021-03-25 17:17:07Running component reader_1, time elapse: 0:00:03
        2021-03-25 17:17:09Running component dataio_0, time elapse: 0:00:06
        2021-03-25 17:17:12Running component intersection_0, time elapse: 0:00:10
        2021-03-25 17:17:17Running component hetero_lr_0, time elapse: 0:00:15
        2021-03-25 17:17:22Running component evaluation_0, time elapse: 0:00:20
        2021-03-25 17:17:26.968 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:129 - Job is success!!! Job id is 202103251717052325809
        2021-03-25 17:17:26.968 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:130 - Total time: 0:00:21
