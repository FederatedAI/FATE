## Hetero Collaborative Memory Network Configuration Usage Guide.

This section introduces the dsl and conf for usage of federated GMF(generalized matrix factorization model).
#### Training Task.
  
#### Training and Evaluation Task.
dsl: test_hetero_cmn_train_then_predict.dsl
runtime_config : test_hetero_cmn_train_then_predict.json

#### test prediction
runtime_config: test_cmn_predict.json
```bash
    # run prediction job
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c examples/federatedrec-examples
/hetero_cmn/test_cmn_predict.json
    # output prediction results
    python {fate_install_path}/fate_flow/fate_flow_client.py -f component_output_data -j {jobId}  -p
 10000 -r guest -cpn hetero_cmn_0 -o {fate_install_path}/logs/{jobId}/guest/10000/hetero_cmn_0
   # get metric results
    python {fate_install_path}/fate_flow/fate_flow_client.py -f component_metric_all -j $jobid -p 10000 -r guest -cpn evaluation_0
```

   
Users can use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.
