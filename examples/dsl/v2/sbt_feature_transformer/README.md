## Sample Weight Configuration Usage Guide.

#### Example Tasks

This section introduces the dsl and conf for different types of tasks.

1.  Hetero SBT + SBT transformer:

    dsl: test_sbt_feat_transformer_dsl_0.json

    runtime_config : test_sbt_feat_transformer_conf_0.json
    
    An Hetero-SBT + SBT transformer, with local baseline comparison.


2. Hetero Fast SBT + SBT transformer:

    dsl: test_sbt_feat_transformer_dsl_1.json

    runtime_config : test_sbt_feat_transformer_conf_1.json
    
    Hetero Fast-SBT + SBT transformer, with local base line comparison and
    transformer model loading.

3. Hetero SBT(Multi) + SBT transformer:

    dsl: test_sbt_feat_transformer_dsl_2.json

    runtime_config : test_sbt_feat_transformer_conf_2.json
    
    Encode samples using multi-sbt
    

Users can use following commands to run the task.

    flow job submit -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use FATE Board to check output. 