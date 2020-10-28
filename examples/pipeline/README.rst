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

Here is a general guide to quick start a FATE job.

1. (optional) create virtual env

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate
      pip install -U pip


2. install fate_client

   .. code-block:: bash

      # this step installs FATE-Pipeline, FATE-Flow CLI v2, and FATE-Flow SDK
      pip install fate_client
      pipeline config --help


3. configure server information

   .. code-block:: bash

      # provide ip and port information
      # optionally, set logs directory
      pipeline config --ip 127.0.0.1 --port 9380

4. upload data with FATE-Pipeline

   .. code-block:: bash

      #  upload data used by demo, optionally provides directory where examples/data locates

      python demo/pipeline-upload.py --base /data/projects/fate

   If upload job is invoked correctly, job id will be printed to terminal and a upload bar is shown.
   If FATE-Board is available, job progress can be monitored on Board as well.

   ::

        UPLOADING:||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||100.00%
        2020-10-16 10:44:26.578 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:125 - Job id is 20201016104426367594590
        Job is still waiting, time elapse: 0:00:01
        Running component upload_0, time elapse: 0:00:03
        2020-10-16 10:44:31.042 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:133 - Job is success!!! Job id is 20201016104426367594590
        2020-10-16 10:44:31.043 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:134 - Total time: 0:00:04

4. run a FATE-Pipeline fit job

   .. code-block:: bash

      python demo/pipeline-quick-demo.py

   This quick demo shows how to build to a heterogeneous SecureBoost job.
   Progress of job execution will be printed as modules run.
   A message indicating final status ("success") will be printed when job finishes.
   The script queries final model information when model training completes.

   ::

        2020-10-27 10:59:43.727 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:125 - Job id is 202010271059435183861
        Job is still waiting, time elapse: 0:00:00
        Running component reader_0, time elapse: 0:00:02
        Running component dataio_0, time elapse: 0:00:05
        Running component intersection_0, time elapse: 0:00:09
        Running component hetero_secureboost_0, time elapse: 0:00:47
        2020-10-27 11:00:31.206 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:133 - Job is success!!! Job id is 202010271059435183861
        2020-10-27 11:00:31.206 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:134 - Total time: 0:00:47

5. (another example) run FATE-Pipeline fit and predict jobs

   .. code-block:: bash

      python demo/pipeline-mini-demo.py

   This script trains a heterogeneous logistic regression model and then runs prediction with the trained model.

   ::

        2020-10-16 13:14:56.316 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:125 - Job id is 20201016131456016425640
        Job is still waiting, time elapse: 0:00:00
        Running component reader_0, time elapse: 0:00:03
        Running component dataio_0, time elapse: 0:00:05
        Running component intersection_0, time elapse: 0:00:10
        Running component hetero_lr_0, time elapse: 0:00:36
        2020-10-16 13:15:33.703 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:133 - Job is success!!! Job id is 20201016131456016425640
        2020-10-16 13:15:33.703 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:134 - Total time: 0:00:37

   Once fit job completes, demo script will print coefficients and other training information of model.

   After having completed the fit job, script will invoke a predict job with the trained model.
   Note that ``Evaluation`` component is added to the prediction workflow. For more information on using
   FATE-Pipeline, please refer to this `guide <../../python/fate_client/pipeline/README.rst>`_.

   ::

        2020-10-16 13:15:34.282 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:125 - Job id is 20201016131533727391641
        Job is still waiting, time elapse: 0:00:02
        Running component reader_1, time elapse: 0:00:05
        Running component dataio_0, time elapse: 0:00:07
        Running component intersection_0, time elapse: 0:00:12
        Running component hetero_lr_0, time elapse: 0:00:17
        Running component evaluation_0, time elapse: 0:00:23
        2020-10-16 13:15:58.206 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:133 - Job is success!!! Job id is 20201016131533727391641
        2020-10-16 13:15:58.207 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:134 - Total time: 0:00:230-10-16 10:46:09.350 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:134 - Total time: 0:00:23
