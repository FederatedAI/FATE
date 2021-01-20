## Data Split Pipeline Example Usage Guide.

#### Example Tasks

This section introduces the Pipeline scripts for different types of tasks.

1. Heterogeneous Data Split Task:

    script: pipeline-hetero-data-split.py
    
    data type: continuous
    
    stratification: stratified by given split points

2. Homogeneous Data Spilt Task:

    script: pipeline-homo-data-split.py
    
    data type: categorical
    
    stratification: stratified by label


3. Homogeneous Data Spilt Task(only validate size specified):

    script: pipeline-homo-data-split-validate.py
    
    data type: categorical
    
    stratification: stratified by label

4. Heterogeneous Data Split Task with Multiple Models:

    script: pipeline-hetero-data-split-multi-model.py
    
    data type: continuous
    
    stratification: stratified by split points

Users can run a pipeline job directly:

    python ${pipeline_script}
