# Pipeline Guide

## Introduction

We provide some example scripts of running FATE jobs with
[FATE-Pipeline](../../../examples/pipeline).

Please refer [here](../../api/fate_client/pipeline.md) for details on
FATE-Pipeline. DSL version of provided Pipeline examples can be found
[here](../../../examples/dsl/v2).

## Quick Start

Here is a general guide to quick start a FATE job. In this guide,
default installation location of FATE is
<span class="title-ref">/data/projects/fate</span>.

1.  (optional) create virtual env
    
    ``` sourceCode bash
    python -m venv venv
    source venv/bin/activate
    pip install -U pip
    ```

2.  install
    fate\_client
    
    ``` sourceCode bash
    # this step installs FATE-Pipeline, FATE-Flow CLI v2, and FATE-Flow SDK
    pip install fate_client
    pipeline init --help
    ```

3.  provide server ip/port information of deployed
    FATE-Flow
    
    ``` sourceCode bash
    # provide real ip address and port info of fate-flow server to initialize pipeline. Typically, the default ip and port are 127.0.0.1:8080.
    pipeline init --ip 127.0.0.1 --port 9380
    # optionally, set log directory of Pipeline
    pipeline init --ip 127.0.0.1 --port 9380 --log-directory {desired log path}
    # show pipeline config details
    pipeline config show
    # check Flow server status according to config
    pipeline config check
    ```

4.  upload data with FATE-Pipeline
    
    Before start a modeling task, the data to be used should be
    uploaded. Typically, a party is usually a cluster which include
    multiple nodes. Thus, when we upload these data, the data will be
    allocated to those nodes.
    
    We have provided an example script to upload data:
    [here](../../../examples/pipeline/demo/pipeline-upload.py).

User may modify file path and table name to upload arbitrary data
following instructions in the script.

> 
> 
> ``` sourceCode python
> #  path to data
> #  default fate installation path
> DATA_BASE = "/data/projects/fate"
> # This is an example for standalone version. For cluster version, you will need to upload your data
> # on each party respectively.
> pipeline_upload.add_upload_data(file=os.path.join(data_base, "examples/data/breast_hetero_guest.csv"),
>                         table_name=dense_data["name"],             # table name
>                         namespace=dense_data["namespace"],         # namespace
>                         head=1, partition=partition)               # data info
> pipeline_upload.add_upload_data(file=os.path.join(data_base, "examples/data/breast_hetero_host.csv"),
>                         table_name=tag_data["name"],
>                         namespace=tag_data["namespace"],
>                         head=1, partition=partition)
> ```
> 
> For a list of available example data and general guide on table
> naming, please refer to this
> [guide](../../../examples/data/README.md).
> 
> ``` sourceCode bash
> #  upload demo data to FATE data storage, optionally provide directory where deployed examples/data locates
> cd /data/projects/fate
> python examples/pipeline/demo/pipeline-upload.py --base /data/projects/fate
> ```
> 
> If upload job is invoked correctly, job id will be printed to terminal
> and an upload bar is shown. If FATE-Board is available, job progress
> can be monitored on Board as
>     well.
> 
>     UPLOADING:||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||100.00%
>     2021-03-25 17:13:21.548 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:121 - Job id is 202103251713214312523
>                        Job is still waiting, time elapse: 0:00:01
>     2021-03-25 17:13:23Running component upload_0, time elapse: 0:00:03
>     2021-03-25 17:13:25.168 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:129 - Job is success!!! Job id is 202103251713214312523
>     2021-03-25 17:13:25.169 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:130 - Total time: 0:00:03
>     UPLOADING:||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||100.00%
>     2021-03-25 17:13:25.348 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:121 - Job id is 202103251713251765644
>                        Job is still waiting, time elapse: 0:00:01
>     2021-03-25 17:13:27Running component upload_0, time elapse: 0:00:03
>     2021-03-25 17:13:29.480 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:129 - Job is success!!! Job id is 202103251713251765644
>     2021-03-25 17:13:29.480 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:130 - Total time: 0:00:04
> 
> > If you would like to change this demo and use your own data, please

5.  run a FATE-Pipeline fit job
    
    ``` sourceCode bash
    cd /data/projects/fate
    python examples/pipeline/demo/pipeline-quick-demo.py
    ```
    
    The details of each step of this demo can be found
    [here](../../../examples/pipeline/demo/pipeline-quick-demo.py).
    
    This quick demo shows how to build to a heterogeneous SecureBoost
    job using uploaded data from previous step. Note that data are
    uploaded to the same machine in the previous step. To run the below
    job with cluster deployment, make sure to first upload data to
    corresponding parties and set role information and job parameters
    accordingly.
    
    Progress of job execution will be printed as modules run. A message
    indicating final status ("success") will be printed when job
    finishes. The script queries final model information when model
    training completes.
    
        2021-11-04 10:12:02.959 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:122 - Job id is 202111041012010774740
                   Job is still waiting, time elapse: 0:00:00
        2021-11-04 10:12:03Running component reader_0, time elapse: 0:00:07
        2021-11-04 10:12:20Running component data_transform_0, time elapse: 0:00:23
        2021-11-04 10:12:36Running component intersection_0, time elapse: 0:00:41
        2021-11-04 10:12:54Running component hetero_secureboost_0, time elapse: 0:01:47
        2021-11-04 10:13:59Running component evaluation_0, time elapse: 0:02:04
        2021-11-04 10:14:17.427 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:130 - Job is success!!! Job id is 202111041012010774740
        2021-11-04 10:14:17.427 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:131 - Total time: 0:02:14

6.  (another example) run FATE-Pipeline fit and predict jobs
    
    ``` sourceCode bash
    cd /data/projects/fate
    python examples/pipeline/demo/pipeline-mini-demo.py
    ```
    
    This [script](../../../examples/pipeline/demo/pipeline-mini-demo.py)
    trains a heterogeneous logistic regression model and then runs
    prediction with the trained model.
    
        2021-11-04 10:07:27.192 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:122 - Job id is 202111041007248446580
                   Job is still waiting, time elapse: 0:00:02
        2021-11-04 10:07:29Running component reader_0, time elapse: 0:00:08
        2021-11-04 10:07:47Running component data_transform_0, time elapse: 0:00:26
        2021-11-04 10:08:03Running component intersection_0, time elapse: 0:00:44
        2021-11-04 10:08:20Running component hetero_lr_0, time elapse: 0:01:29
        2021-11-04 10:09:07.846 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:130 - Job is success!!! Job id is 202111041007248446580
        2021-11-04 10:09:07.846 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:131 - Total time: 0:01:40
               
    Once fit job completes, demo script will print coefficients and
    training information of model.
    
    After having completed the fit job, script will invoke a predict job
    with the trained model. Note that `Evaluation` component is added to
    the prediction workflow. For more information on using
    FATE-Pipeline, please refer to this
        [guide](../../api/fate_client/pipeline.md).
    
        2021-11-04 10:09:10.899 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:122 - Job id is 202111041009082730800
                   Job is still waiting, time elapse: 0:00:08
        2021-11-04 10:09:19Running component reader_1, time elapse: 0:00:14
        2021-11-04 10:09:37Running component data_transform_0, time elapse: 0:00:32
        2021-11-04 10:09:53Running component intersection_0, time elapse: 0:00:50
        2021-11-04 10:10:11Running component hetero_lr_0, time elapse: 0:01:10
        2021-11-04 10:10:32Running component evaluation_0, time elapse: 0:01:30
        2021-11-04 10:10:52.516 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:130 - Job is success!!! Job id is 202111041009082730800
        2021-11-04 10:10:52.516 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:131 - Total time: 0:01:41
