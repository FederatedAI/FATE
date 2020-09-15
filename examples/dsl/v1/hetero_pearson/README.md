## Hetero Pearson Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Training Task.

- dsl: test_conf.json
- runtime_config : test_conf.json
 
    
Users can use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

#### Params

1. column_indexes: -1 or list of int. If -1 provided, all columns are used for calculation. If a list of int provided, columns with given indexes are used for calculation.
2. column_names: names of columns use for calculation.

Notice that, if both params are provided, the union of columns indicated are used for calculation.  
