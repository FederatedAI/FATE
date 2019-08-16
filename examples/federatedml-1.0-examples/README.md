## Instructions for building models

There are three config files need to be prepared to build a algorithm model in FATE. After that, starting a task will be as simple as running the command below.

### Step1: Define upload data config file

To make FATE be able to use your data, you need to upload them. Thus, a upload-data conf is needed. A sample file named "upload_data.json" has been provided in **dsl_test** folder.

#### Field Specification
1. file: file path
2. head: Specify whether your data file include a header or not
3. partition: Specify how many partitions used to store the data
4. table_name & namespace: Indicators for stored data table.
5. work_mode: Indicate if using standalone version or cluster version. 0 represent for standalone version and 1 stand for cluster version.

### Step2: Define your modeling task structure

Practically, when building a modeling task, several components might be involved, such as data_io, feature_engineering, algorithm_model, evaluation as so on. However, the combination of these components would differ from task to task. Therefore, a convenient way to freely combine these components would be a critical feature.

Currently, FATE provide a kind of domain-specific language(DSL) to define whatever structure you want. The components are combined as a Directed Acyclic Graph(DAG) through the dsl config file. The usage of dsl config file is as simple as defining a json file.

The DSL config file will define input data and(or) model as well as output data and(or) model for each component. The downstream components take output data and(or) model of upstream components as input. In this way, a DAG can be constructed by the config file.

We have provided several example dsl files in **dsl_test** folder. Here is some points may worth paying attention to.

#### Field Specification
1. component_name: key of a component. This name should end with a "_num" such as "_0", "_1" etc. And the number should start with 0. This is used to distinguish multiple same kind of components that may exist.
2. work_mode: Indicate if using standalone version or cluster version. 0 represent for standalone version and 1 stand for cluster version.
3. module: Specify which component use. This field should be one of the algorithm modules FATE supported.
   FATE-1.0 supports 11 usable algorithm module.

   > DataIO: transform input-data into Instance Object for later components
   > Intersection: find the intersection of data-set different parties, mainly used in hetero scene modeling.
   > FederatedSample: sample module for making data-set balance, supports both federated and standalone mode.
   > FeatureScale: module for feature scaling and standardization.
   > HeteroFeatureBinning: With binning input data, calculates each column's iv and woe and transform data according to the binned information.
   > HeteroFeatureSelection: feature selection module, supports both federated and standalone.
   > OneHotEncoder: feature encoding module, mostly used to encode the binning result.
   > HeteroLR: hetero logistic regression module.
   > HomoLR: homo logistic regression module.
   > HeteroSecureBoost: hetero secure-boost module.
   > Evaluation: evaluation module. support metrics for binary, multi-class and regression.

4. input: There are two type of input, data and model.
    1. data: There are three possible data_input type:
        1. data: typically used in data_io, feature_engineering modules and evaluation.
        2. train_data: Used in homo_lr, hetero_lr and secure_boost. If this field is provided, the task will be parse as a **fit** task
        3. eval_data: If train_data is provided, this field is optional. In this case, this data will be used as validation set. If train_data is not provided, this task will be parse as a **predict** or **transform** task.
    2. model: There are two possible model-input type:
        1. model: This is a model input by same type of component, used in prediction\transform stage. For example, hetero_binning_0 run as a fit component, and hetero_binning_1 take model output of hetero_binning_0 as input so that can be used to transform or predict.
        2. isometric_model: This is used to specify the model input from upstream components, only used by HeteroFeatureSelection module in FATE-1.0. HeteroFeatureSelection can take the model output of HetereFeatureBinning and
5. output: Same as input, two type of output may occur which are data and model.
    1. data: Specify the output data name
    2. model: Specify the output model name

6. need_deploy: true or false. This field is used to specify whether the component need to deploy for online inference or not. This field just use for online-inference dsl deduction.

### Step3: Define configuration for each specific component.
This config file is used to config parameters for all components among every party.
1. initiator: Specify the initiator's role and party id
2. role: Indicate all the party ids for all roles.
3. role_parameters: Those parameters are differ from roles and roles are defined here separately. Please note each parameter are list, each element of which corresponds to a party in this role.
4. algorithm_parameters: Those parameters are same among all parties are here.

### Step4: Start modeling task

#### 1. Upload data
Before starting a task, you need to load data among all the data-providers. To do that, a load_file config is needed to be prepared.  Then run the following command:

> python ${your_install_path}/fate_flow/fate_flow_client.py -f upload -c dsl_test/upload_data.json

Here is an example of configuring upload_data.json:
```
    {
      "file": "examples/data/breast_b.csv",
      "head": 1,
      "partition": 10,
      "word_mode": 0,
      "table_name": "breast_b",
      "namespace": "hetero_breast"
    }
```

We use **breast_b** & **hetero_breast** as guest party's table name and namespace. To use default runtime conf, please set host party's name and namespace as **breast_a** & **hetero_breast** and upload the data with path of  **examples/data/breast_a.csv**

To use other data set, please change your file path and table_name & namespace. Please do not upload different data set with same table_name and namespace.

Note: This step is needed for every data-provide node(i.e. Guest and Host).

#### 2. Start your modeling task
In this step, two config files corresponding to dsl config file and component config file should be prepared. Please make sure the table_name and namespace in the conf file match with upload_data conf. should be Then run the following command:

> python ${your_install_path}/fate_flow/fate_flow_client.py -f submitJob -d hetero_logistic_regression/test_hetero_lr_job_dsl.json -c hetero_logistic_regression/test_hetero_lr_job_conf.json

#### 3. Check log files
Now you can check out the log in the following path: ${your_install_path}/logs/{your jobid}.


### Step5: Check out Results
FATE now provide "FATE-BOARD" for showing modeling log-metrics and evaluation results.

Use your browser to open a website: http://{Your fate-board ip}:{your fate-board port}/index.html#/history.

<div style="text-align:center", align=center>
<img src="../image/JobList.png" alt="samples" width="500" height="250" /><br/>
Figure 1： Federated HomoLR Principle</div>

There will be all your job history list here. Your latest job will be list in the first page. Use JOBID to find out the modeling task you want to check.

<div style="text-align:center", align=center>
<img src="../image/JobOverview.png" alt="samples" width="500" height="250" /><br/>
Figure 1： Job Overview</div>

In the task page, all the components will be shown as a DAG. We use different color to indicate their running status.
1. Green: run success
2. Blue: running
3. Gray: Waiting
4. Red: Failed.

 You can click each component to get their running parameters on the right side. Below those parameters, there exist a **View the outputs** button. You may check out model output, data output and logs for this component.

<div style="text-align:center", align=center>
<img src="../image/ComponentOutput.png" alt="samples" width="500" height="250" /><br/>
Figure 1： Job Overview</div>

If you want a big picture of the whole task, there is a **dashboard** button on the right upper corner. Get in the Dashboard, there list three windows showing different information.

<div style="text-align:center", align=center>
<img src="../image/DashBoard.png" alt="samples" width="500" height="250" /><br/>
Figure 1： Job Overview</div>

1. Left window: showing data set used for each party in this task.
2. Middle window: Running status or progress of the whole task
3. Right window: DAG of components.

### Step6: Check out Logs

After you submit a job, you can find your job log in ${Your install path}/logs/${your jobid}

The logs for each party is collected separately and list in different folders. Inside each folder, the logs for different components are also arranged in different folders. In this way, you can check out the log more specifically and get useful detailed  information.


## Quick Start

We have provided a python script for quick starting modeling task.

### Command Line Interface

- command: python quick_run.py
- parameter:
    * -w  --work_mode: work mode, 1 represent for cluster version while 0 stand for standalone version, default 0, Optional
    * -r  --role: start role, needed only in cluster version, Optional
    * -a  --algorithm: Selection algorithm. Support hetero_lr, homo_lr, hetero_secureboost, default hetero_lr, Optional
    * -gid --guest_id: Only needed in cluster version, Optional
    * -hid --host_id: Only needed in cluster version, Optional
    * -aid --arbiter_id: Only needed in cluster version, Optional

- description: quick start a job.


### Standalone Version
1. Start standalone version hetero-lr task (default)
> python quick_run.py

```
stdout:{
    "data": {
        "board_url": "http://localhost:8080/index.html#/dashboard?job_id=20190815211211735986134&role=guest&party_id=10000",
        "job_dsl_path": "${your install path}/jobs/20190815211211735986134/job_dsl.json",
        "job_runtime_conf_path": "/data/projects/fate/python/jobs/20190815211211735986134/job_runtime_conf.json",
        "model_info": {
            "model_id": "arbiter-10000#guest-10000#host-10000#model",
            "model_version": "20190815211211735986134"
        }
    },
    "jobId": "20190815211211735986134",
    "meta": null,
    "retcode": 0,
    "retmsg": "success"
}


Please check your task in fate-board, url is : http://localhost:8080/index.html#/dashboard?job_id=20190815211211735986134&role=guest&party_id=10000
The log info is located in ${your install path}/examples/federatedml-1.0-examples/../../logs/20190815211211735986134
```

Then you are supposed to see the above output. You can view the job on the url above or check out the log through the log file path.

2. Start standalone version homo-lr task
> python quick_run.py -a homo_lr

3. Start standalone version hetero-secureboost task
> python quick_run.py -a hetero_secureboost

### Cluster Version

1. Host party:
> python quick_run.py -w 1 -r host

This is just uploading data

2. Guest party:
> python quick_run.py -w 1 -r guest -gid guest_id -hid host_id -aid arbiter_id -a algorithm

The config files that generated is stored in a new created folder named **user_config**

### Some common usages of fate flow
#### 1.How to get the output data of each component:

 >cd {your_fate_path}/fate_flow
 
 >python fate_flow_client.py -f component_output_data -j $jobid -p $party_id -r $role -cpn $component_name -o $output_dir

jobid:the task jobid you want to get.

party_id:your mechine's party_id, such as 10000

role:"guest" or "host" or "arbiter"
 
component_name: the component name which you want to get, such as component_name "hetero_lr_0" in {your_fate_path}/examples/federatedml-1.0-examples/hetero_logistic_regression/test_hetero_lr_train_job_dsl.json

output_dir: the output directory

#### 2.How to get the output model of each component
 
 >python fate_flow_client.py -f component_output_model -j $jobid -p $party_id -r $role -cpn $component_name


#### 3.How to get the logs of task

 >python fate_flow_client.py -f job_log -j $jobid -o $output_dir
 
#### 4.How to stop the job

 > python fate_flow_client.py -f stop_job -j $jobid

#### 5.How to query job current status

 > python fate_flow_client.py -f query_job -j $jobid -p party_id -r role

#### 6.How to get the job runtime configure
 > python fate_flow_client.py -f job_config -j $jobid -p party_id -r role -o $output_dir

#### 7.How to download a table which has been uploaded before
 > python fate_flow_client.py -f download -n table_namespace -t table_name -w work_mode -o save_file
 
 work_mode: will be 0 for standalone or 1 for cluster, which depend on what you set in upload config