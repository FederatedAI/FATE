Pipeline Examples
=================

Introduction
-------------

We provide some example scripts of running
FATE jobs with `FATE-Pipeline <../../python/fate_client/README.rst>`_.

Please refer to the document linked above for details on FATE-Pipeline and FATE-Flow CLI v2.
DSL version of provided Pipeline examples can be found `here <../dsl/v2>`_.


Quick start
-----------

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
        2020-10-16 10:44:31.042 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:133 - Job is complete!!! Job id is 20201016104426367594590
        2020-10-16 10:44:31.043 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:134 - Total time: 0:00:04

4. run FATE-Pipeline fit and predict jobs

   .. code-block:: bash

      python demo/pipeline-mini-demo.py

   As job runs, progress of job execution will also be printed as modules are executed.
   A message indicating final status ("complete") will also be printed at the end of the job.

   ::

        2020-10-16 10:44:47.492 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:125 - Job id is 20201016104447123987592
        Job is still waiting, time elapse: 0:00:00
        Running component reader_1, time elapse: 0:00:03
        Running component reader_0, time elapse: 0:00:06
        Running component dataio_0, time elapse: 0:00:09
        Running component dataio_1, time elapse: 0:00:11
        Running component intersection_1, time elapse: 0:00:17
        Running component intersection_0, time elapse: 0:00:22
        Running component hetero_lr_0, time elapse: 0:01:00
        2020-10-16 10:45:49.165 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:133 - Job is complete!!! Job id is 20201016104447123987592
        2020-10-16 10:45:49.165 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:134 - Total time: 0:01:01

   Once fit job completes, demo script will print coefficients & validation metrics of result model.

   After having completed a fit job, script will invoke a predict job with the model from previous fit job.
   Note how only deployed modules are included in the predict job workflow. For more information on using
   FATE-Pipeline, please refer to this `guide <../../python/fate_client/pipeline/README.rst>`_.

   ::

        2020-10-16 10:45:49.765 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:125 - Job id is 20201016104549184225593
        Job is still waiting, time elapse: 0:00:03
        Running component reader_2, time elapse: 0:00:05
        Running component dataio_0, time elapse: 0:00:08
        Running component intersection_0, time elapse: 0:00:13
        Running component hetero_lr_0, time elapse: 0:00:18
        2020-10-16 10:46:09.349 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:133 - Job is complete!!! Job id is 20201016104549184225593
        2020-10-16 10:46:09.350 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:134 - Total time: 0:00:19
