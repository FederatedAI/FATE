### FATE-Flow Client Command Line Interface

#### Job

- command: fate_flow_client.py -f submit_job
- parameter:
    * -c  --config: runtime conf path, Required
    * -d  --dsl: dsl path, Required
- description: submit job


- command: fate_flow_client.py -f stop_job
- parameter:
    * -j  --job_id: job id, Required
- description: stop job 


- command: fate_flow_client.py -f query_job
- parameter:
    * -j  --job_id: job id, Required
    * -p  --party_id: party id, Optional
    * -r  --role : role, Optional
    * -s  --status: status, Optional
- description: query information about this job


- command: fate_flow_client.py -f job_config 
- parameter:
    * -j  --job_id: job id, Optional
    * -p  --party_id: party id, Optional
    * -r  --role : role, Optional
    * -o  --output_path: config output path, Required
- description: download the configuration of this job


- command: fate_flow_client.py -f job_log
- parameter: 
    * -j  --job_id: job id, Optional
    * -o  --output_path: config output path, Required
- description: download the log of this job


- command: fate_flow_client.py -f query_task 
- parameter:
    * -j  --job_id: job id, Optional
    * -p --party_id: party id, Optional
    * -r  --role : role, Optional
    * -cpn --component_name: component name, Optional
    * -s  --status: status, Optional
- description: query information about this task


#### Tracking

- command: fate_flow_client.py -f component_parameters 
- parameter:
    * -j --job_id: job id, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required
    * -cpn --component_name: component name, Required
- description: query the parameters of this component


- command: fate_flow_client.py -f component_metric_all
- parameter:
    * -j --job_id: job id, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required
    * -cpn --component_name: component name, Required
- description: query all metric data 


- command: fate_flow_client.py -f component_metrics
- parameter:
    * -j --job_id: job id, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required
    * -cpn --component_name: component name, Required
- description: query metrics


- command: fate_flow_client.py -f component_output_model
- parameter:
    * -j --job_id: job id, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required
    * -cpn --component_name: component name, Required
- description: query this component model meta


- command: fate_flow_client.py -f component_output_data 
- parameter:
    * -j --job_id: job id, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required
    * -cpn --component_name: component name, Required
    * -o  --output_path: config output path, Required
- description: download the output data of this component


#### DataAccess

- command: fate_flow_client.py -f download
- parameter:
    * -n --namespace: namespace, Required
    * -t --table_name: table name, Required
    * -o --output_path: output path, Required
- description: download data


- command: fate_flow_client.py -f upload 
- parameter:
    * -c --config: config path, Required
- description: upload data


#### Table

- command: fate_flow_client.py -f table_info
- parameter:
    * -n --namespace: namespace, Optional
    * -t  --table_name: table name, Optional
- description: query table information


#### Model

- command: fate_flow_client.py -f load
- description: load mode

  
- command: fate_flow_client.py -f online
- description: publish_online


- command: fate_flow_client.py -f version 
- parameter:
    * -n --namespace: namespace, Required
- description: query model version history
