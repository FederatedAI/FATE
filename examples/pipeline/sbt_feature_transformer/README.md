## Sample Weight Configuration Usage Guide.

#### Example Tasks

This section introduces the dsl and conf for different types of tasks.

1.  Hetero SBT + SBT transformer:
    
    script: pipeline-sbt-transformer.py
    
    An Hetero-SBT + SBT transformer, with local baseline comparison.


2. Hetero Fast SBT + SBT transformer:
       
    script: pipeline-sbt-transformer-fast-sbt.py
    
    Hetero Fast-SBT + SBT transformer, with local base line comparison and
    transformer model loading.

3. Hetero SBT(Multi) + SBT transformer:
       
    script: pipeline-sbt-transformer-multi.py
    
    Encode samples using multi-sbt
    

Users can use following commands to run pipeline job directly.

    python ${pipeline_script}

After having finished a successful training task, you can use FATE Board to check output. 