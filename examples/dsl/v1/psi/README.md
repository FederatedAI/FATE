## PSI Configuration Usage Guide.

This section introduces the dsl and conf relationships for usage.

#### Training Task.

1. PSI:  

    example-data: 
    
    (1) guest: expect: breast_homo_guest.csv; actual: breast_homo_host.csv
    (2) host: expect: breast_homo_guest.csv; actual: breast_homo_host.csv
     
    dsl: psi_cpn_dsl.json  
    
    runtime_config: psi_cpn_conf.json  
     

Note: users should upload the data described above with specified table name and namespace in the runtime_config, 
then use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config{ -d ${dsl}
