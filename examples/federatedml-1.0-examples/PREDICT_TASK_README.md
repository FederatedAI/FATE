## Instructions for using training models to predict

In order to use trained model to predict. The following several steps are needed.

### Step1: Train Model. Please check [here](./README.md)

Pay attention to following points to enable predicting:
1. you should add/modify "need_deploy" field for those modules that need to deploy in predict stage. All modules have set True as their default value except FederatedmSample and Evaluation, which typically will not run in predict stage. The "need_deploy" field is True means this module should run a "fit" process and the fitted model need to be deployed in predict stage.

2. Except setting those model as "need_deploy", they should also config to have a model output except Intersect module. Only in this way can fate-flow store the trained model and make it usable in inference stage.

3. Get training model's model_id and model_version. There are two ways to get this.
    
   a. After submit a job, there will be some model information output in which "model_id" and "model_version" are our interested field.
   b. Beside that, you can also obtain these information through the following command directly:
       
       python ${your_fate_install_path}/fate_flow/fate_flow_client.py -f job_config -r guest -p ${guest_partyid}  -o ${job_config_output_path}

       where
       $guest_partyid is the partyid of guest (the party submitted the job)
       $job_config_output_path: path to store the job_config
       
      After that, a json file including model info will be download to ${job_config_output_path}/model_info.json in which you can find "model_id" and "model_version".
      
### Step2: define your predict config.

This config file is used to config parameters for predicting.

1. initiator: Specify the initiator's role and party id, it should be same with training process.
2. job_parameters: 
    
    work_mode: cluster or standalone, it should be same with training process.
    model_id\model_version: model indicator which mentioned in Step1.
    job_type: type of job. In this case, it should be "predict".
3. role: Indicate all the party ids for all roles, it should be same with training process.
4. role_parameters: Set parameters for each roles. In this case, the "eval_data", which means data going to be predicted, should be filled for both Guest and Host parties.

### Step3. Start your predict task

After complete your predict configuration, run the following command.
    
    python ${your_fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${predict_config}
    
### Step4: Check out Running State. 

Running State can be check out in FATE_board whose url is http://${fate_board_ip}:${fate_board_port}/index.html#/details?job_id=${job_id}&role=guest&party_id=${guest_partyid}

where

${fate_board_ip}\${fate_board_port}: ip and port to deploy the FATE board module.

${job_id}: the predict task's job_id.

${guest_partyid}: the guest party id

You can also checkout job status by fate_flow in case without FATE_board installed.

The following command is used to query job status such as running, success or fail.

    python /data/projects/fate/python/fate_flow/fate_flow_client.py -f query_job -j {job_id} -r guest
   
### Step5: Download Predicting Results.

Once predict task finished, the first 100 records of predict result are available in FATE-board. You can also download all results through the following command.
  
    python ${your_fate_install_path}/fate_flow/fate_flow_client.py -f component_output_data -j ${job_id} -p ${party_id} -r $role} -cpn ${component_name} -o ${predict_result_output_dir}

    where
    ${jobIid}: predict task's job_id
    ${party_id}: the partyid of current user.
    ${role}: the role of current user. Please keep in mind that host users are not supposed to get predict results in heterogeneous algorithm.
    ${component_name}: the component who has predict results
    ${predict_result_output_dir}: the directory which use download the predict result to.

