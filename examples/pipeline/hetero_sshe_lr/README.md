## Hetero SSHE Logistic Regression Configuration Usage Guide.

This section introduces the Pipeline scripts for different types of tasks.

#### Example Tasks

1. Train_task:
    script: pipeline-hetero-lr-normal.py

2. LR  Compute Loss:
    script: pipeline-hetero-compute-loss.py

3. Cross Validation Task(with fold history data output of predict score):
    script: pipeline-hetero-lr-cv.py

4. One vs Rest(OVR) Task:
    script: pipeline-hetero-lr-ovr.py

5. LR with validation:
    script: pipeline-hetero-lr-with-validate.py

6. LR with Warm start task:
    script: pipeline-hetero-lr-warm-start.py

7. LR with Encrypted Reveal in Host task:
    script: pipeline-hetero-lr-encrypted-reveal-in-host.py

8. LR L1 penalty task:
    script: pipeline-hetero-lr-l1.py

9. OVR LR with Encrypted Reveal in Host task:
    script: pipeline-hetero-lr-ovr-encrypted-reveal-in-host.py

10. LR OVR None-penalty task:
    script: pipeline-hetero-lr-ovr-none-penalty.py

11. LR OVR L1 penalty task:
    script: pipeline-hetero-lr-ovr-l1.py

12. LR with Large Init Weight:
    script: pipeline-hetero-lr-large-init-w-compute-loss.py

13. LR without intercept:
    script: pipeline-hetero-lr-not-fit-intercept.py

14. LR Compute Loss without reveal:
    script: pipeline-hetero-lr-compute-loss-not-reveal.py

15. LR with Sample Weight:
    script: pipeline-hetero-lr-sample-weight.py


Users can run a pipeline job directly:

    python ${pipeline_script}
