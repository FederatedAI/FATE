## Quick Start

We supply standalone and cluster mode of running examples for Feature Selection algorithm.

### 1. Run Standalone Version

In standalone mode, role guest and host are invoked. You can start them through following steps:

cd your_install_path/examples/hetero_feature_selection/

sh run_feature_selection_standalone.sh

After doing these two steps, you can wait for the result or go to check some logs for this task. The log files is: your_install_path/logs/feature_selection_{timestamp}


### 2. Run Cluster Version
In cluster version, you can use task-manager which is a tool help you start all the parties easily. The task manager client is located at : /arch/task_manager/task_manager_client.py.

1. Before starting a cluster version task, you need to load data among all the data-providers. To do that, you need to edit a load_file config.

 Two example files for host and guest are prepared in ./conf folder:

        guest: conf/load_file_tm_guest.json.

       "file": $FATE_install_path/examples/data/breast_b.csv

       host: conf/load_file_tm_host.json.

       "file": $FATE_install_path/examples/data/breast_a.csv

Then run the following command:

   guest:
   > python $FATE_install_path/arch/task_manager/task_manager_client.py -f upload -c conf/load_file_tm_guest.json

   host:
   > python $FATE_install_path/arch/task_manager/task_manager_client.py -f upload -c conf/load_file_tm_host.json


2. Then, you need to edit a config file for all the parties. A sample config file has been provided in this folder. As the sample file shows, the parameters that are different among all the parties should be set in role_parameters respectively. On the other hand, those parameters that are same should be put in algorithm_parameters.

You should re-write the configure of role guest "train_input_table" using "table_name" you have got in guest's load data, and "train_input_namespace" using "namespace" you have got. The same as the configure of role host using "table_name" and "namespace" you got after host load data.

3. After finish editing, you can run the following command to start the task:

> python $FATE_install_path/arch/task_manager/task_manager_client.py -f workflow -c conf/test_feature_selection_workflow.json

After running this command, a jobid will be generated automatically for you.


### 3. Check log files

4. Now you can check out the log in the following path: your_install_path/logs/{your jobid}.

### 4. More functions of task-manager

There are a couple of more functions that task-manager has provided. Please check [here](../../arch/task_manager/README.md)

### 5. Some error you may encounter
While run standalone version, you may get info "task failed, check nohup in current path". please check the nohup files to see if there exists any errors.

While run cluster version, if you find not {jobid} fold in your_install_path/logs, please check your_install_path/jobs/{jobid}/upload/std.log or your_install_path/jobs/{jobid}/guest/std.log to find if there exist any error

Check logs/{jobid}/status_tracer_decorator.log file if there exist any error during these task
