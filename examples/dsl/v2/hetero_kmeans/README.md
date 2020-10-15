## Hetero Kmeans Configuration Usage Guide.

#### Example Tasks

This section introduces the dsl and conf for different types of tasks.

1. Train Task:

    dsl: test_hetero_kmeans_train_dsl.json

    runtime_config : test_hetero_kmeans_train_conf.json

2. Validate Task (with early-stopping parameters specified):

    dsl: test_hetero_kmeans_validate_dsl.json

    runtime_config : test_hetero_kmeans_validate_conf.json

3. Multi-host Train Task:

    dsl: test_hetero_kmeans_multi_host_dsl.json

    conf: test_hetero_kmeans_multi_host_conf.json

4. With Feature-engineering Task:

    dsl: test_hetero_kmeans_with_feature_engineering_dsl.json

    conf: test_hetero_kmeans_with_feature_engineering_conf.json



Users can use following commands to run a task.

    bash flow job submit -c ${runtime_config} -d ${dsl}
