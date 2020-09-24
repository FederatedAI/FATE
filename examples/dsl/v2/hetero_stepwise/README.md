## Hetero Stepwise Configuration Usage Guide.

#### Example Tasks

This section introduces the dsl and conf for different types of tasks.

1. Logistic Regression Model:  
    example-data: 
    
        (1) guest: breast_hetero_mini_guest.csv      
        (2) host: breast_hetero_mini_host.csv 
         
    dsl: test_hetero_stepwise_lr_dsl.json  
    
    runtime_config: test_hetero_stepwise_lr_conf.json
     
2. Linear Regression Model:  
    example-data: 
    
        (1) guest: motor_hetero_mini_guest.csv
        (2) host: motor_hetero_mini_host.csv  
        
    dsl: test_hetero_stepwise_linr_dsl.json  
    
    runtime_config: test_hetero_stepwise_linr_conf.json
   
3. Poisson Regression:  
    example-data: 
    
        (1) guest: dvisits_hetero_guest.csv
        (2) host: dvisits_hetero_host.csv  
        
    dsl: test_hetero_stepwise_poisson_dsl.json  
    
    runtime_config: test_hetero_stepwise_poisson_conf.json
    
   
Users can use following commands to run a task.
   
    bash flow job submit -c ${runtime_config{ -d ${dsl}
