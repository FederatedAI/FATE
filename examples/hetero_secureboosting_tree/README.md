## Quick Start

We supply standalone and cluster mode of running examples for HeteroSecureBoostingTree algorithm. 

### 1. Run Standalone Version

In standalone mode, a host and a guest are invoked. You cdan start the two through following step


1. Open two terminal(or two session)

2. In each of the two terminal. go to examples/hetero_secureboosting_tree/

3. Start host by 
   > sh run_host.sh host_runtime_conf jobid

4. Start guest by

   > sh run_guest.sh guest_runtime_conf jobid

jobid is a string which host and guest should have the same jobid, please remember that jobid should be different for different task,
host_runtime_conf or guest_runtime_conf template is under example/hetero_secureboosting_tree/host(guest)_runtime_conf.json.

Or you can simply execute run_secureboosting_standalone.sh, which will run host and guest in the background and print out some infomation, like logfiles path.

> sh run_secureboosting_standalone.sh

After doing all above steps, you can check logs of this task in logs/$jobid.

### 2. Run Cluster Version

In cluster mode, you need to use task-manager service to submit task. We provide a task-manager client to access to this service easily.
What's more, in cluster mode, only the party guest need to submit the job once both parties loads their data to FATE storage service.

Formally, you can follow steps to run jobs in cluster mode.

1. Loading data.
   
   a. go to examples/hetero_secureboosting_tree/
   
   b. modify conf
   
       guest: conf/load_file_tm_guest.json. 
       
       "file": $FATE_install_path/examples/data/breast_b.csv
       
       host: conf/load_file_tm_host.json. 
       
       "file": $FATE_install_path/examples/data/breast_a.csv
       
   c. both guest and host run the following command separately. After executing, a json-like information prints out, 
      please pick up jobid and check out logs in $FATE_install_path/jobs/$jobid/upload/std.log to ensure upload data successfully
   
   guest: 
   > python $FATE_install_path/arch/task_manager/task_manager_client.py -f upload -c conf/load_file_tm_guest.json 
   
   host: 
   > python $FATE_install_path/arch/task_manager/task_manager_client.py -f upload -c conf/load_file_tm_host.json 
       
2. Running task. 

    After both parties uploaded their data, guest should run the following command. 
    
    > python $FATE_install_path/arch/task_manager/task_manager_client.py -f workflow -c conf/hetero_secureboost_tm_submit_conf.json
    
    jobid will also print out in screen once the command is executed. 
    
    The task is "cross_validation", if you want to run "train", please have a look at example in examples/hetero_logistci_regression
    
Please pay attention that if you want to run other datasets, please execute the following commands and run step 1 and 2 again, otherwise you may use the same table_name again which might raise some errors during running.
   
    guest: modify conf/load_file_tm_guest.json
       
        "file": $guest_dataset_path
        "table_name": $new_guest_data_table
       
    host: modify conf/load_file_tm_host.json
       
        "file": $host_dataset_path
        "table_name": $new_host_data_table
     
    host sync table name to guest, then guest modify conf/hetero_secureboost_tm_submit_conf.json
    guest should modify data_input_table of guest WorkflowParam and host WorkflowParam.
        
        ...
        "role_parameters": {
            ...
            "guest": {
                "WorkFlowParam": {
                    "data_input_table": [$new_guest_data_table],
                    ...
                 } 
                 ...     
            },
            ...
            "host": {
                "WorkFlowParam": {
                    "data_input_table": [$new_host_data_table],
                    ...
                }
                ...
            }
        }
        ...
          
    
### 3. Check log files.

Now you can check out the log in the following path: $FATE_install_path/logs/$jobid
    
### 4. More functions of task-manager

There are a couple of more functions that task-manager has provided. Please check [here](../../arch/task_manager/README.md)

### 5. Some error you may encounter
1. While run standalone version, you may get info *"task failed, check nohup in current path"*. please check the nohup files to see if there exists any errors.

2. While run cluster version, if you find not $jobid fold in  *FATE_install_path/logs*, please check  *FATE_install_path/jobs/{jobid}/upload/std.log* or *FATE_install_path/jobs/$jobid/guest/std.log* to find if there exist any error

3. Check logs/$jobid/status_tracer_decorator.log file if there exist any error during these task
 

