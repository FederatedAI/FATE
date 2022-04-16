## Homo Logistic Regression Pipeline Example Usage Guide.

#### Example Tasks

This section introduces the Pipeline scripts for different types of tasks.

1. Cross Validation Task:

    script: pipeline-homo-lr-cv.py

2. Multi-host Task:

    script: pipeline-homo-lr-multi-host.py

3. Train Task:

    script: pipeline-homo-lr-train.py

4. Single Predict Task:
    
    script: pipeline-homo-lr-eval.py

5. Train with validate Task:  
    
    script: pipeline-homo-lr-train-eval.py

6. Multi-Class Train Task:  

   script: pipeline-homo-lr-one-vs-all.py
    
7. Multi-Class Train Task With Paillier:  

   script: pipeline-homo-lr-one-vs-all-encrypted-host.py

Users can run a pipeline job directly:

    python ${pipeline_script}
