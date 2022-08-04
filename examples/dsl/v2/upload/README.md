## Upload Configuration Usage Guide.

#### Example Tasks

This section introduces the dsl and conf for different types of tasks.

1. Upload Task:

    runtime_config : upload_conf.json

2. Upload Tag Task:

    runtime_config : upload_tag_conf.json

3. Upload Task with Anonymous Header:

    runtime_config : upload_anonymous_conf.json

Users can use following commands to run the task.

    flow data upload -c ${runtime_config}
