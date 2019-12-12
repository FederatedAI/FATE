from test_example import submit
import json
f=open("standalone-task.json")
config=json.loads(f.read())
result={}
s = submit.Submitter().set_fate_home("/usr/fate/standalone-fate-master-1.1").set_work_mode(0)
for key in config.keys():
    flag = config[key]['host'].find("@")
    if flag != -1:                                        #cluster task
        a = config[key]['host'].split("@", 1)
        s.run_upload(data_path=a[0], remote_host=a[1])
    else:                                                  #standalone task
        s.run_upload(config[key]['host'])
    s.run_upload(config[key]['guest'])
    job_id = s.submit_job(config[key]['conf'], config[key]['dsl'])
    result[key]=s.await_finish(job_id)
with open("result.txt", "w") as f:
    sys.stdout = f
    print(result)