## Quick Start

We supply standalone and cluster mode of running examples for HeteroLogisticRegression algorithm.

### 1. Run Standalone Version

In standalone mode, role host, guest and arbiter are invoked. You can start them through following steps:

> cd $FATE_install_path/examples/hetero_logistic_regression/

> sh run_logistic_regression_standalone.sh 

After doing these two steps, you can wait for the result or go to check some logs for this task. The log files is: $FATE_install_path/logs/hetero_logistic_regression_example_standalone_${timestamp}


### 2. Run Cluster Version
In cluster version, you can use task-manager which is a tool help you start all the parties easily.
> cd $FATE_install_path/examples/hetero_logistic_regression/

#### load data
Before starting a cluster version task, you need to load data among all the data-providers. We have prepared some example data for hetero-lr. You should edit the file conf/load_file_tm_guest and conf/load_file_tm_host to make sure the data file path is right. 
Then upload data using:

In role guest:
>  python $FATE_install_path/arch/task_manager/task_manager_client.py -f upload -c conf/load_file_tm_guest.json

In role host:
>  python $FATE_install_path/arch/task_manager/task_manager_client.py -f upload -c conf/load_file_tm_host.json

After load data, you can get "table_name" and "namespace" which have been edit in the configure file and please be attention to that the "table_name" should be different for each upload.


#### run task
Then, you need to edit a config file for role guest. A sample config file like *test_hetero_lr_workflow.json* has been provided in this folder. As the sample file shows, the parameters that are different among all the parties should be set in role_parameters respectively. On the other hand, those parameters that are same should be put in algorithm_parameters.


You should re-write the configure of  role guest "train_input_table" using "table_name" you have got in guest's load data, and "train_input_namespace" using "namespace" you have got. The same as the configure of  role host using "table_name" and "namespace" you got after host load data.

If you want to predict, please write the configure of "predict_input_table" and "predict_input_namespace". The configure of "model_table", "predict_output_table" and "evaluation_output_table" in role guest or host should be different for each task.


After finish editing, you can run the following command to start the task:

> python $FATE_install_path/arch/task_manager/task_manager_client.py -f workflow -c conf/test_hetero_lr_workflow.json

After running this command, a jobid will be generated automatically for you.

### 3. Check log files

4. Now you can check out the log in the following path: $FATE_install_path/logs/${your jobid}.

### 4. More functions of task-manager

There are a couple of more functions that task-manager has provided. Please check [here](../../arch/task_manager/README.md)

### 5. Some error you may encounter
1. While run standalone version, you may get info *"task failed, check nohup in current path"*. please check the nohup files to see if there exists any errors.
2. While run cluster version, if you find not ${jobid} fold in  *$FATE_install_path/logs*, please check  *$FATE_install_path/jobs/${jobid}/upload/std.log* or *$FATE_install_path/jobs/${jobid}/guest/std.log* to find if there exist any error
3. Check logs/${jobid}/status_tracer_decorator.log file if there exist any error during these task
 