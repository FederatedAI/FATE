## Hetero Stepwise Pipeline Example Usage Guide.

#### Example Tasks

This section introduces the Pipeline scripts for different types of tasks.

1. Logistic Regression Model:  
    example-data: 
    
        (1) guest: breast_hetero_mini_guest.csv      
        (2) host: breast_hetero_mini_host.csv 
         
    script: pipeline-hetero-stepwise-lr.py
     
2. Linear Regression Model:  
    example-data: 
    
        (1) guest: motor_hetero_mini_guest.csv
        (2) host: motor_hetero_mini_host.csv  
        
    script: pipeline-hetero-stepwise-linr.py
   
3. Poisson Regression:  
    example-data: 
    
        (1) guest: dvisits_hetero_guest.csv
        (2) host: dvisits_hetero_host.csv  
        
    script: pipeline-hetero-stepwise-poisson.py
    
   
Users can run a pipeline job directly:

    python ${pipeline_script}
