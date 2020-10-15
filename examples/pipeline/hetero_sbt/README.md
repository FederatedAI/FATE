## Hetero SecureBoost Pipeline Example Usage Guide.

#### Example Tasks

This section introduces the Pipeline scripts for different types of tasks.

1. Train Binary Classification Task:

    script: pipeline-hetero-sbt-binary.py

2. Train Binary Classification Task with prediction:

    script: pipeline-hetero-sbt-binary-with-predict.py

3. Train Multi-class Classification Task:

    script: pipeline-hetero-sbt-multi.py

4. Train Regression Task:

    script: pipeline-hetero-sbt-regression.py
    
5. Training With Complete Secure Activated:

    script: pipeline-hetero-sbt-binary-complete-secure

6. Training With Early-stop Activated:

    script: pipeline-hetero-sbt-with-early-stop

7. Train Binary Classification With Missing Features:

    script: pipeline-hetero-sbt-with-missing-value

8. Train Binary Classification Task With Cross-Validation:

    script: pipeline-hetero-sbt-binary-cv.py

9. Train Multi-class Classification Task Cross-Validation:

    script: pipeline-hetero-sbt-multi-cv.py

10. Train Regression Task Cross-Validation:

    script: pipeline-hetero-sbt-regression-cv.py
    
11. Train With Multi-host:

    script: pipeline-hetero-sbt-regression-multi-host.py

Users can run a pipeline job directly:

    python ${pipeline_script}
