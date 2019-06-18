# Task Manager(TM)

To make FATE easy to use, we provide a tool call task manager. With usage of it, you can start both standalone and cluster version task in one terminal. Besides, you can set parameters of all parties in one config files. In one word, you can do everything in one terminal with one config file.
## Con
### Execution Target 
Local party configuration.
```json
"local":{
  "party_id": 9999,
  "role": "guest"
}
```
### All Party
```json
"role": {
  "guest": [9999],
  "host": [10000],
  "arbiter": [10000]
},
```
### Automatically generate table information
```json
"gen_table_info": true,
"data_type": "train_input"
```
If `table_name` and `namespace` is null, as the same time, `gen_table_info` is true and `data_type` is not null.TM will use `local`,`role`,`scene_id` and `data_type` configuration to generate table_name and namespace. 
## Examples

We have list several example config files and the TM client script here so that you can easily start to find out the usage of TM. To start task with TM, configuration is (maybe the only) important steps. More example config files has been list in : **arch/task_manager/example_conf**. You use -f represent for function to indicate the function you want.

 More detail information has been listed as follow.

### Load File

Before we start to construct a model, loading data is a necessary step. An example of load file configuration has been provided. Please check out hte load_file.json for detail information. A critical key you need to notify is that you should specify "work_mode" in the config file. 0 means standalone version and 1 represent for clustering mode.

After you have set up everything, you can start to load file as simple as:

>  python task_manager_client.py -f upload -c upload_file.json

### Import ID Library from Local File

TM provide a function for you to import id only from a file instead of loading whole data set. The example config file is located in **arch/task_manager/example_conf**. TM will load the 0th column as id column. So if it is not your case, please do some pre-processing work for the data. The call statement is similar:

> python task_manager_client.py -f upload -c example_conf/import_id_library_host.json

### Download ID Library

> python task_manager_client.py -f download -c example_conf/download_id_library_host.json

### Start workflow

This method is use to start a workflow task. You can config all parties configuration in one config file. For those parameters that are different in each party, you set in **role_parameters**. On the other hand, set parameters that are same among all parties in **algorithm_parameters**. The call statement is as follow:

> python task_manager_client.py -f workflow -c test_hetero_lr_workflow.json

### Query Job Status

After you commit a task. You can check the status of task by:

> python task_manager_client.py -f jobStatus -j 20190416_194554_10000_67

where -j means the jobid of a task

### Get Runtime conf

You can also acquire the runtime configuration by:

> python task_manager_client.py -f workflowRuntimeConf -j 20190417_162654_10000_1

This statement will copy the config file to current directory.

### Download Data use command line configuration

You are able to download data from DTable and save as a file. The call statement is :

> python task_manager_client.py -f download -n '50000_guest_9999_9999&10000_train_input' -t '0b9e5612603911e9a888acde48001122' -o train_input_table -p 9999

where -n is namespace, -t is table name, -o is downloaded file name and -p is the party id.

### Query job queue

> python task_manager_client.py -f queueStatus