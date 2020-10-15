## Hetero Poisson Regression Pipeline Example Usage Guide.

#### Example Tasks

This section introduces the Pipeline scripts for different types of tasks.

1. Train & Predict Task:

    script: pipeline-hetero-poisson.py
   
2.  Validate Task (with early-stopping parameter specified):

    script: pipeline-hetero-poisson-validate.py
  
3. Train Task with Sparse Data:
    
    script: pipeline-hetero-poisson-sparse.py

5. Cross Validation Task:

    script: pipeline-hetero-poisson-cv.py


Users can run a pipeline job directly:

    python ${pipeline_script}
