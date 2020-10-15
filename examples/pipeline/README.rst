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

   ::

       UPLOADING:||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||100.00%
       2020-10-15 21:27:48.889 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:125 - Job id is 20201015212748684073567
       Job is still waiting, time elapse: 0:00:00
       Running component upload_0, time elapse: 0:00:03
       2020-10-15 21:27:52.238 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:133 - Job is complete!!! Job id is 20201015212748684073567
       2020-10-15 21:27:52.238 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:134 - Total time: 0:00:03
       UPLOADING:||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||100.00%
       2020-10-15 21:27:52.450 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:125 - Job id is 20201015212752242218568
       Job is still waiting, time elapse: 0:00:02
       Running component upload_0, time elapse: 0:00:08
       2020-10-15 21:28:00.945 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:133 - Job is complete!!! Job id is 20201015212752242218568
       2020-10-15 21:28:00.946 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:134 - Total time: 0:00:08

4. run FATE-Pipeline fit and predict jobs

   .. code-block:: bash

      python demo/pipeline-mini-demo.py

   As job runs, progress of job execution will also be printed as modules are executed.
   A message indicating final status ("complete") will also be printed at the end of the job.

   ::
        2020-10-15 21:29:15.388 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:125 - Job id is 20201015212915057697569
        Job is still waiting, time elapse: 0:00:00
        Running component reader_1, time elapse: 0:00:02
        Running component reader_0, time elapse: 0:00:05
        Running component dataio_0, time elapse: 0:00:08
        Running component dataio_1, time elapse: 0:00:11
        Running component intersection_1, time elapse: 0:00:16
        Running component intersection_0, time elapse: 0:00:21
        Running component hetero_lr_0, time elapse: 0:00:44
        2020-10-15 21:30:00.402 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:133 - Job is complete!!! Job id is 20201015212915057697569
        2020-10-15 21:30:00.402 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:134 - Total time: 0:00:45

   Once fit job completes, demo script will print coefficients of result model:

   ::

        {
            "best_iteration": 1,
            "coef": {
                "x0": 0.16838478819471825,
                "x1": 0.05698790256408079,
                "x2": 0.27731104498859516,
                "x3": -0.01270254817885296,
                "x4": -0.015313578736286405,
                "x5": 0.7898765686442109,
                "x6": 0.027314342334738492,
                "x7": 0.8612655665270401,
                "x8": 0.036424897390035474,
                "x9": 0.411767957613962
            },
            "intercept": 0.9625916968372231,
            "is_converged": false,
            "one_vs_rest": false,
            "validation_metrics": {
                "auc": [
                    0.08073040536969506,
                    0.08309550235188416,
                    0.08507742719729403
                ],
                "ks": [
                    0.0,
                    0.0,
                    0.0
                ]
            }
        }


   After having completed a fit job, script will invoke a predict job with the model from previous fit job.
   Note how only deployed modules are included in the predict job workflow. For more information on using
   FATE-Pipeline, please refer to this `guide <../../python/fate_client/pipeline/README.rst>`_.

   ::

        2020-10-15 21:30:00.967 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:125 - Job id is 20201015213000425859570
        Job is still waiting, time elapse: 0:00:03
        Running component reader_2, time elapse: 0:00:05
        Running component dataio_0, time elapse: 0:00:08
        Running component intersection_0, time elapse: 0:00:13
        Running component hetero_lr_0, time elapse: 0:00:18
        2020-10-15 21:30:20.306 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:133 - Job is complete!!! Job id is 20201015213000425859570
        2020-10-15 21:30:20.306 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:134 - Total time: 0:00:19


