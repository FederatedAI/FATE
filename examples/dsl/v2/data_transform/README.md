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

Users can use following commands to run a task.

    flow job submit -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.
