# DSL version 1 predict tutorial
This documentation will give a breif tutorial of how to run a predict task after a trainning task.
We will take hetero-secureboost as an example.
	
## Submit a training task
We can start a training job by submitting conf & dsl through flow client,
Here we submit a hetero-secureboost binary classification task, whose conf and dsl are in [hetero secureboost example 
folder.](../../dsl/v1/hetero_secureboost)

    >> flow job submit -c ./examples/dsl/v1/hetero_secureboost/test_secureboost_train_binary_conf.json -d ./examples/dsl/v1/hetero_secureboost/test_secureboost_train_dsl.json
    >> {
            "data": {
                "board_url": "http://127.0.0.1:8080/index.html#/dashboard?job_id=2020103015490073208469&role=guest&party_id=10000",
                "job_dsl_path": "fate/jobs/2020103015490073208469/job_dsl.json",
                "job_runtime_conf_path": "/fate/jobs/2020103015490073208469/job_runtime_conf.json",
                "logs_directory": "/fate/logs/2020103015490073208469",
                "model_info": {
                    "model_id": "guest-10000#host-10000#model",  <<- model_id needed for prediction tasks
                    "model_version": "2020103015490073208469"  <<- model_version needed for prediction tasks
                }
            },
            "jobId": "2020103015490073208469",
            "retcode": 0,
            "retmsg": "success"
        }

Then we can get a return message contains model_id and model_version. They are needed for making a prediction conf.

## Retrieve model_id and model_version
Forget to save model_id and model_version in the returned message? No worry. 
You can query the corresponding model_id and model_version of a job using the "flow job config" command.

    >> flow job config -j 2020103015490073208469 -r guest -p 9999 -o ./
    >> {
            "data": {
                "job_id": "2020103015490073208469",
                "model_info": {
                    "model_id": "guest-10000#host-10000#model", <<- model_id needed for prediction tasks
                    "model_version": "2020103015490073208469" <<- model_version needed for prediction tasks
                },
                "train_runtime_conf": {}
            },
            "retcode": 0,
            "retmsg": "download successfully, please check /fate/job_2020110210415854150330_config directory",
            "directory": "/fate/job_2020110210415854150330_config"
        }

## Make a predict conf
We can modify existing predict conf by replacing model_id, model_version and data set name with yours to make a new 
predict conf.
Here we replace model_id and model_version in [predict conf](../../dsl/v1/hetero_secureboost/test_predict_conf.json) 
with model_id and model_version returned by training job.

    {
        "initiator": {
            "role": "guest",
            "party_id": 10000
        },
        "job_parameters": {
            "work_mode": 0,
            "job_type": "predict",
            "model_id": "guest-10000#host-10000#model",  <<-- to replace 
            "model_version": "2020103015490073208469"  <<-- to replace
        },
        "role": {
            "guest": [
                10000
            ],
            "host": [
                10000
            ]
        },
        "role_parameters": {
            "guest": {
                "args": {
                    "data": {
                        "eval_data": [
                            {
                                "name": "breast_hetero_guest",   <<-- you can set predict dataset here
                                "namespace": "experiment"
                            }
                        ]
                    }
                }
            },
            "host": {
                "args": {
                    "data": {
                        "eval_data": [
                            {
                                "name": "breast_hetero_host",  <<-- you can set predict dataset here
                                "namespace": "experiment"
                            }
                        ]
                    }
                }
            }
        }
    }

## Submit a predict job
Then we can submit a new predict job:
    
    >> flow job submit -c ./examples/dsl/v1/hetero_secureboost/test_predict_conf.json 