# FATE-Flow Client Command Line Interface

## Usage
```bash
python fate_flow_client.py -f $command
```



## JOB_OPERATE

#### submit_job
- description: submit a pipeline job
- parameter:
    * -c  --config: runtime conf path, Required
    * -d  --dsl: dsl path, Required
```bash
python fate_flow_client.py -f submit_job -c examples/test_hetero_lr_job_conf.json -d examples/test_hetero_lr_job_dsl.json 
```
    
    
        
#### stop_job
- description: cancel job or stop job 
- parameter:
    * -j  --job_id: job id, Required
```bash
python fate_flow_client.py -f stop_job -j $job_id
```
    
    
    
#### query_job
- description: query job information by filters
- parameter:
    * -j  --job_id: filter by job id, Optional
    * -r  --role : filter by role, Optional
    * -p  --party_id: filter by party id, Optional
    * -s  --status: filter by status, Optional
```bash
python fate_flow_client.py -f query_job -j $job_id
```

#### clean_job
- description: clean processor,data table and metric data
- parameter:
    * -j  --job_id: filter by job id, Optional
    * -r  --role : filter by role, Optional
    * -p  --party_id: filter by party id, Optional
    * -cpn --component_name: component name, Optional
```bash
python fate_flow_client.py -f clean_job -j $job_id
```

#### data_view_query
-description: query data view information by filters
- parameter:
    * -j  --job_id: filter by job id, Optional
    * -r  --role : filter by role, Optional
    * -p  --party_id: filter by party id, Optional
    * -s  --status: filter by status, Optional
```bash
python fate_flow_client.py -f data_view_query -j $job_id
```


## JOB
#### job_config
- description: download the configuration of this job
- parameter:
    * -j  --job_id: job id, Required
    * -r  --role : role, Required
    * -p  --party_id: party id, Required
    * -o  --output_path: config output directory path, Required
```bash
python fate_flow_client.py -f job_config -j $job_id -r $role -p $party_id -o $output_path
```


#### job_log
- description: download the log of this job
- parameter: 
    * -j  --job_id: job id, Required
    * -o  --output_path: config output directory path, Required
```bash
python fate_flow_client.py -f job_log -j $job_id -o $output_path
```



## TASK_OPERATE
#### query_task
- description: query task information by filters
- parameter:
    * -j  --job_id: filter by job id, Optional
    * -cpn --component_name: filter by component name, Optional
    * -r  --role : filter by role, Optional
    * -p --party_id: filter by party id, Optional
    * -s  --status: filter by status, Optional
```bash
python fate_flow_client.py -f query_task -j $job_id 
```



## TRACKING
#### component_parameters
- description: query the parameters of this component
- parameter:
    * -j --job_id: job id, Required
    * -cpn --component_name: component name, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required
```bash
python fate_flow_client.py -f component_parameters -j $job_id -r $role -p $party_id -cpn $component_name
```


#### component_metric_all
- description: query all metric data 
- parameter:
    * -j --job_id: job id, Required
    * -cpn --component_name: component name, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required
```bash
python fate_flow_client.py -f component_metric_all -j $job_id -r $role -p $party_id -cpn $component_name
```


#### component_metrics
- description: query the list of metrics
- parameter:
    * -j --job_id: job id, Required
    * -cpn --component_name: component name, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required
```bash
python fate_flow_client.py -f component_metrics -j $job_id -r $role -p $party_id -cpn $component_name
```



#### component_output_model
- description: query this component model
- parameter:
    * -j --job_id: job id, Required
    * -cpn --component_name: component name, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required
```bash
python fate_flow_client.py -f component_output_model -j $job_id -r $role -p $party_id -cpn $component_name
```



#### component_output_data
- description: download the output data of this component
- parameter:
    * -j --job_id: job id, Required
    * -cpn --component_name: component name, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required
    * -o  --output_path: config output path, Required
    * -limit  --limit: Limit quantity, Optional
```bash
python fate_flow_client.py -f component_output_data -j $job_id -r $role -p $party_id -cpn $component_name -o $output_path
```


#### component_output_data_table
- description: view table name and namespace
- parameter:
    * -j --job_id: job id, Required
    * -cpn --component_name: component name, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required
```bash
python fate_flow_client.py -f component_output_data_table -j $job_id -r $role -p $party_id -cpn $component_name 
```


### DATA

#### download
- description: download table
- parameter:
    * -c --config: config path, Required
```bash
python fate_flow_client.py -f download -c examples/download_guest.json
```



#### upload
- description: upload table
- parameter:
    * -c --config: config path, Required
    * -drop --drop: Operations before file upload, Optional 
```bash
python fate_flow_client.py -f upload -c examples/upload_guest.json 
python fate_flow_client.py -f upload -c examples/upload_guest.json -drop 0(or1)
```



#### upload_history
- description: upload table history
- parameter:
    * -j --job_id: job id, Optional
    * -limit --limit: Limit quantity, Optional
```bash
python fate_flow_client.py -f upload_history -j $job_id
python fate_flow_client.py -f upload_history -limit 5
```



### Table

#### table_info
- description: query table information
- parameter:
    * -n --namespace: namespace, Required
    * -t --table_name: table name, Required
```bash
python fate_flow_client.py -f table_info -n $namespace -t $table_name
```



#### table_delete
- description: delete table 
- parameter:
    * -n --namespace: namespace, Optional
    * -t  --table_name: table name, Optional
    * -j --job_id: job id, Optional
    * -cpn --component_name: component name, Optional
    * -r --role: role, Optional
    * -p --party_id: party id, Optional
```bash
python fate_flow_client.py -f table_delete -n $namespace -t $table_name
python fate_flow_client.py -f table_delete -j $job_id
```




### Model

#### load
- description: load model.
- parameter:
    * -c --config: config path, Required
```bash
python fate_flow_client.py -f load -c $conf_path
```



#### bind
- description: bind model.
- parameter:
    * -c --config: config path, Required
```bash
python fate_flow_client.py -f bind -c $conf_path
```



#### store
- description: store model
- parameter:
    * -c --config: config path, Required
```bash
python fate_flow_client.py -f store -c $conf_path
```


#### restore
- description: restore mode
- parameter:
    * -c --config: config path, Required
```bash
python fate_flow_client.py -f restore -c $conf_path
```


#### export
- description: export model
- parameter:
    * -c --config: config path, Required
```bash
python fate_flow_client.py -f export -c $conf_path
```


#### import
- description: import model
- parameter:
    * -c --config: config path, Required
```bash
python fate_flow_client.py -f import -c $conf_path
```