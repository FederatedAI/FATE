## Hetero SecureBoost Pipeline Example Usage Guide.

#### Example Tasks

This section introduces the Pipeline scripts for different types of tasks.

1. Train on Binary Label:

   script: test_hetero_sbt_binary.py

2. Cross Validation on Binary Label:

   script: test_hetero_sbt_binary_cv.py

3. Warm Start on Binary Label:

   script: test_hetero_sbt_binary_warm_start.py

4. Train on Multi-class Label:

   script: test_hetero_sbt_multi.py

5. Train on Continuous Label:

   script: test_hetero_sbt_regression.py

Users can run a pipeline job directly:

    python ${pipeline_script}