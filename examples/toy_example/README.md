## Quick Start

We supply standalone and cluster mode of running examples for Toy Example. 

### 1. Run Standalone Version

In standalone mode, a host and a guest are invoked. You cdan start the two through following step


1. Open two terminal(or two session)

2. In each of the two terminal. go to examples/toy_example/

3. Start host by 
   > sh run_host.sh host_runtime_conf.json jobid

4. Start guest by

   > sh run_guest.sh guest_runtime_conf jobid

jobid is a string which host and guest should have the same jobid, please remember that jobid should be different for different task,
host_runtime_conf or guest_runtime_conf template is under example/toy_example/conf/host(guest)_runtime_conf.json.

Or you can simply execute run_toy_example_standalone.sh, which will run host and guest in the background and print out some infomation, like logfiles path.

> sh run_toy_example_standalone.sh

After doing all above steps, you can check logs of this task in logs/$jobid.

### 2. Run Cluster Version

In cluster mode, you need to use task-manager service to submit task. We provide a task-manager client to access to this service easily.

1. Running toy. 

    After both parties uploaded their data, guest should run the following command. 
    
    > python $FATE_install_path/arch/task_manager/task_manager_client.py -f workflow -c conf/toy_example_tm_submit.json
    
    jobid will also print out in screen once the command is executed. 
    
### 3. Check log files.

Now you can check out the log in the following path: $FATE_install_path/logs/$jobid
    
### 4. More functions of task-manager

There are a couple of more functions that task-manager has provided. Please check [here](../../arch/task_manager/README.md)

### 5. Some error you may encounter
1. While run standalone version, you may get info *"task failed, check nohup in current path"*. please check the nohup files to see if there exists any errors.

2. While run cluster version, if you find not $jobid fold in  *FATE_install_path/logs*, please check  *FATE_install_path/jobs/{jobid}/upload/std.log* or *FATE_install_path/jobs/$jobid/guest/std.log* to find if there exist any error

3. Check logs/$jobid/status_tracer_decorator.log file if there exist any error during these task
 

