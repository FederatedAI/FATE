### FATE-Flow Client Command Line Interface

#### Job

- command: fate_flow_client.py -f submit_job
- endpoint: v1/job/submit
- parameter:
    * -c  --config: runtime conf path
    * -d  --dsl: dsl path


- command: fate_flow_client.py -f stop_job
- endpoint: v1/job/stop
- parameter:
    * -j  --job_id: job id


- command: fate_flow_client.py -f query_job
- endpoint: v1/job/query
- parameter:
    * -j  --job_id: job id
    * -p  --party_id: party id
    * -r  --role : role
    * -s  --status: status


- command: fate_flow_client.py -f job_config 
- endpoint: v1/job/config
    * -j  --job_id: job id
    * -p --party_id: party id
    * -r  --role : role
    * -o  --output_path : output directory


- command: fate_flow_client.py -f job_log 
- endpoint: v1/job/log
    * -j  --job_id: job id
    * -o  --output_path : output directory


- command: fate_flow_client.py -f query_task 
- endpoint: v1/job/task/query
    * -j  --job_id: job id
    * -p --party_id: party id
    * -r  --role : role
    * -cpn --component_name: component name
    * -s  --status: status


#### Tracking

- command: fate_flow_client.py -f component_parameters 
- endpoint: v1/tracking/component/parameters
- parameter:
    * -j --job_id: job id
    * -r --role: role
    * -p --party_id: party id
    * -cpn --component_name: component name


- command: fate_flow_client.py -f component_metric_all
- endpoint: v1/tracking/component/metric/all
- parameter:
    * -j --job_id: job id
    * -r --role: role
    * -p --party_id: party id
    * -cpn --component_name: component name


- command: fate_flow_client.py -f component_metrics
- endpoint: v1/tracking/component/metrics
- parameter:
    * -j --job_id: job id
    * -r --role: role
    * -p --party_id: party id
    * -cpn --component_name: component name


- command: fate_flow_client.py -f component_output_model
- endpoint: v1/component/output/model
- parameter:
    * -j --job_id: job id
    * -r --role: role
    * -p --party_id: party id
    * -cpn --component_name: component name


- command: fate_flow_client.py -f component_output_data 
- endpoint: v1/component/output/data
- parameter:
    * -j --job_id: job id
    * -r --role: role
    * -p --party_id: party id
    * -cpn --component_name: component name
    * -o  --output_path : output directory


#### DataAccess

- command: fate_flow_client.py -f download
- endpoint: v1/data/download
- parameter:
    * -n --namespace: namespace
    * -t  --table_name: table name
    * -o --output_path: output path


- command: fate_flow_client.py -f upload 
- endpoint: v1/data/upload
- parameter:
    * -c --config: config path


#### Table

- command: fate_flow_client.py -f table_info
- endpoint: v1/table/table_info
- parameter:
    * -n --namespace: namespace
    * -t  --table_name: table name


#### Model

- command: fate_flow_client.py -f load
- endpoint: v1/mode/load
  

- command: fate_flow_client.py -f online
- endpoint: v1/mode/online


- command: fate_flow_client.py -f version 
- endpoint: v1/mode/version
- parameter:
    * -n --namespace: namespace
