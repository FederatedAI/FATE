## Label Transform Pipeline Example Usage Guide.

#### Example Tasks

This section introduces the Pipeline scripts for different types of tasks.

1. Label Transform Task(default mapping, no difference):

    script: pipeline-label-transform.py

2. Label Transform Task with Encoder(with given label mapping):

    script: pipeline-label-transform-encoder.py

3. Label Transform Task with Encoder(without label list):

    script: pipeline-label-transform-encoder-without-label-list.py

4. Label Transform Task with Partially-specified Encoder(without label list):

    script: pipeline-label-transform-encoder-partial.py


Users can run a pipeline job directly:

    python ${pipeline_script}
