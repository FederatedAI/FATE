## PSI Configuration Usage Guide.

This section introduces the dsl and conf relationships for usage.

#### Example Task.

1. PSI:  

    example-data: 
    
    (1) guest: expect: breast_homo_guest.csv; actual: breast_homo_host.csv
    (2) host: expect: breast_homo_guest.csv; actual: breast_homo_host.csv
     
    dsl: psi_cpn_dsl.json  
    
    runtime_config: psi_cpn_conf.json  
     

Users can use following commands to run a task.

    flow job submit -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.
