## Instructions for using training models to predict

In order to use trained model to predict. The several steps need.

### Step1: training your model. Please check [here](./README.md)

Pay attention to following points to enable predicting:
1. you should add/modify "need_deploy" field if need. All modules except FederatedmSample and Evaluation, 
their "need_deploy" field's value are true in default(You can check this under [this fold](../../federatedml/conf/setting_conf).

2. Please confirm again that all modules that whose "need_deploy" field are true should have be forward process, 
means that they are running "fit" process, and have output model.

3. Get training model's model_id and model_version. There are two ways to get this.
    
   a. once you submit the jobs, something will be printed out, "model_id" and "model_version" are exactly what we want.
   b. if you forget to mark down the information the screent prints out, down be worried, use the following command can also help you.
       
       python ${your_fate_install_path}/fate_flow/fate_flow_client.py -f job_config -r guest -p ${guest_partyid}  -o $job_config_output_path
       
       $guest_partyid is the partyid of guest (the party submitted the job)
       $job_config_output_path: path to store the job_config
       
      After running the command, you can find "model_id" and "model_version" in $job_config_output_path/model_info.json
      
### Step2: define your predict config.

This config file is used to config parameters for predicting.

1. initiator: Specify the initiator's role and party id, it should be the same with training process.
2. job_parameters: 
    
    work_mode: cluster or standalone, it should be the same with training process.
    model_id\model_version: the two keywords mean which model uses to predict. They can be find the Step1.
    job_type: the type of job, it should be "predict"
3. role: Indicate all the party ids for all roles, it should be the same of training process.
4. role_parameters: your should fill the "eval_data" of role "guest" and "host".

### Step3. Start your predict task

After complete your predict configuration, run the below command.
    
    python ${your_fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${predict_config}
    
### Step4: Check out Running State. 

Running State can be check out in FATE_board, url is http://${fate_board_ip}:${fate_board_port}/index.html#/details?job_id=${job_id}&role=guest&party_id=${guest_partyid}

${fate_board_ip}\${fate_board_port}: ip and port to deploy the FATE board module.

${job_id}: the predict task' job_id.

${guest_partyid}: the party guest's party_id

You can also checkout job status by fate_flow if you don't want to install FATE_board, execute the following command will be helpful.

    python /data/projects/fate/python/fate_flow/fate_flow_client.py -f query_job -j {job_id} -r guest
   
### Step5: Download Predicting Results.

Once predict task is finished, you can checkout the result in FATE_board, but it only shows the first 100 records.
If you want to download all results, execute the following command please.
  
    python ${your_fate_install_path}/fate_flow/fate_flow_client.py -f component_output_data -j ${job_id} -p ${party_id} -r $role} -cpn ${component_name} -o ${predict_result_output_dir}
    
    ${jobIid}: predict task's job_id
    ${party_id}: the partyid who wants to get its output result
    ${role}: the role who wants to get its output result, mostly it should be guest( host have predict output result only in Homo Logistic Regression)
    ${component_name}: the component who has predict results
    ${predict_result_output_dir}: the directory use to download the predict result.

