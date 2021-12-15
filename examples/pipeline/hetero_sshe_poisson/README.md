## Hetero SSHE Poisson Regression Pipeline Example Usage Guide.

#### Example Tasks

This section introduces the Pipeline scripts for different types of tasks.

1. Train & Predict Task:

    script: pipeline-hetero-poisson.py

2. Warm-start Task:

    script: pipeline-hetero-poisson-warm-start.py

3. Validate Task (with early-stopping parameters specified):

    script: pipeline-hetero-poisson-validate.py

4. Cross Validation Task:

    script: pipeline-hetero-poisson-cv.py

5. Train & Predict Task without Revealing Loss:

    script: pipeline-hetero-poisson-compute-loss-not-reveal.py

6. Train Task with encrypted_reveal_in_host:
    
    script: pipeline-hetero-poisson-encrypted-reveal-in-host.py


Users can run a pipeline job directly:

    python ${pipeline_script}
