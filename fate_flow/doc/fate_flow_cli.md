## FATE-Flow Client Command Line Interface

### Job

- **command: fate_flow_client.py -f submit_job**
- description: submit a pipeline job
- parameter:
    * -c  --config: runtime conf path, Required
    * -d  --dsl: dsl path, Required
    
    
    
- **command: fate_flow_client.py -f stop_job**
- description: stop job 
- parameter:
    * -j  --job_id: job id, Required
    
    
    
- **command: fate_flow_client.py -f query_job**
- description: query job information by filters
- parameter:
    * -j  --job_id: filter by job id, Optional
    * -r  --role : filter by role, Optional
    * -p  --party_id: filter by party id, Optional
    * -s  --status: filter by status, Optional



- **command: fate_flow_client.py -f job_config**
- description: download the configuration of this job
- parameter:
    * -j  --job_id: job id, Required
    * -r  --role : role, Required
    * -p  --party_id: party id, Required
    * -o  --output_path: config output directory path, Required



- **command: fate_flow_client.py -f job_log**
- description: download the log of this job
- parameter: 
    * -j  --job_id: job id, Required
    * -o  --output_path: config output directory path, Required



- **command: fate_flow_client.py -f query_task**
- description: query task information by filters
- parameter:
    * -j  --job_id: filter by job id, Optional
    * -cpn --component_name: filter by component name, Optional
    * -r  --role : filter by role, Optional
    * -p --party_id: filter by party id, Optional
    * -s  --status: filter by status, Optional

### Tracking

- **command: fate_flow_client.py -f component_parameters** 
- description: query the parameters of this component
- parameter:
    * -j --job_id: job id, Required
    * -cpn --component_name: component name, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required



- **command: fate_flow_client.py -f component_metric_all**
- description: query all metric data 
- parameter:
    * -j --job_id: job id, Required
    * -cpn --component_name: component name, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required



- **command: fate_flow_client.py -f component_metrics**
- description: query the list of metrics
- parameter:
    * -j --job_id: job id, Required
    * -cpn --component_name: component name, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required



- **command: fate_flow_client.py -f component_output_model**
- description: query this component model
- parameter:
    * -j --job_id: job id, Required
    * -cpn --component_name: component name, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required



- **command: fate_flow_client.py -f component_output_data**
- description: download the output data of this component
- parameter:
    * -j --job_id: job id, Required
    * -cpn --component_name: component name, Required
    * -r --role: role, Required
    * -p --party_id: party id, Required
    * -o  --output_path: config output path, Required

### DataAccess

- **command: fate_flow_client.py -f download**
- description: download table
- parameter:
    * -w --work mode: work mode, Required
    * -n --namespace: namespace, Required
    * -t --table_name: table name, Required
    * -o --output_path: output path, Required



- **command: fate_flow_client.py -f upload**
- description: upload table
- parameter:
    * -c --config: config path, Required

### Table

- **command: fate_flow_client.py -f table_info**
- description: query table information
- parameter:
    * -n --namespace: namespace, Required
    * -t  --table_name: table name, Required

### Model

- **command: fate_flow_client.py -f load**
- description: load model.Since the model is not well controlled here, the interface here is not implemented.



- **command: fate_flow_client.py -f online**
- description: publish model online.Since the model is not well controlled here, the interface here is not implemented.



- **command: fate_flow_client.py -f version**
- description: query model version history
- parameter:
    * -n --namespace: namespace, Required
