## Hetero SSHE Linear Regression Pipeline Example Usage Guide.

#### Example Tasks

This section introduces the Pipeline scripts for different types of tasks.

1. Train & Predict Task:

    script: pipeline-hetero-linr.py

2. Warm-start Task:

    script: pipeline-hetero-linr-warm-start.py

3. Validate Task (with early-stopping parameters specified):

    script: pipeline-hetero-linr-validate.py

4. Cross Validation Task:

    script: pipeline-hetero-linr-cv.py

5. Train & Predict Task without Revealing Loss:

    script: pipeline-hetero-linr-compute-loss-not-reveal.py

6. Train Task with encrypted-reveal-in-host:

    script: pipeline-hetero-linr-encrypted-reveal-in-host.py

7. Train Task with large init weight:
    
    script: pipeline-hetero-linr-large-init-w-compute-loss.py

7. Train Task with Weighted Instances:
    
    script: pipeline-hetero-linr-sample-weight.py

Users can run a pipeline job directly:

    python ${pipeline_script}
