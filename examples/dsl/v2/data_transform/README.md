## Data Transform Configuration Usage Guide.

#### Example Tasks

This section introduces the dsl and conf for different types of tasks.

1. Dense InputFormat Task:

    example-data: (1) guest: breast_hetero_guest.csv (2) host: breast_hetero_host.csv  
    
    dsl: test_data_transform_dsl.json

    runtime_config : test_data_transform_dense_conf.json

2. TagValue InputFormat Task:
    
    example-data: (1) guest: breast_hetero_guest.csv  
                  (2) host: tag_value_1000_140.csv  
    
    dsl: test_data_transform_dsl.json

    runtime_config: test_data_transform_tag_value_conf.json

3. Input Data With Missing Value:

    example-data: (1) guest: ionosphere_scale_hetero_guest.csv  (2) host: ionosphere_scale_hetero_host.csv 
    
    dsl: test_data_transform_dsl.json

    runtime_config : test_data_transform_missing_fill_conf.json 

4. SVM-Light InputFormat Task:  

    examples-data: (1) guest: svmlight_guest.csv  (2) host: svmlight_host.csv  
    
    dsl: test_data_transform_validate_dsl.json 
    
    runtime_config: test_data_transform_svmlight.json
    
5. Dense InputFormat With MatchID Task:

    example-data: sample with Task.1, but should upload with extend_sid=True  
    
    dsl: test_data_transform_match_id_dsl.json   
    
    runtime_config: test_data_transform_dense_match_id_conf.json  
    
6. TagValue InputFormat With MatchID Task:

    example-data: sample with Task.1, but should upload with extend_sid=True  
    
    dsl: test_data_transform_match_id_dsl.json   
    
    runtime_config: test_data_transform_tag_value_match_id_conf.json  


Users can use following commands to run a task.

    flow job submit -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.
