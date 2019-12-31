Instructions of test tools
=================
1.Use 
------
Execute commands<br>
cd examples/test<br>
python run_test.py default_env.json<br>

run_test.py  search and execute tasks defined by users. <br>
default_env.json environment configs based on users' running environment. <br>

Optional parameters<br>
 "-o", "--output", "file to save result, defaults to `test_result`" <br>
 "-e", "--error", "file to save error" <br>
 "-i", "--interval", "check job status every i seconds, defaults to 3" <br>
 "--skip_data", "skip data upload, used to be false if not use <br>
 
 mutually_exclusive_group include: <br>
 "-d", "--dir", "dir to find testsuites", <br>
 "-s", "--suite","a single testsuite to run" <br>


2.Tips
------
If '-d' or '-s' is not given,the script will execute the tasks defined in task files from examples/federatedml-1.x-examples folder with a "testsuite.json" suffix.<br>
If there is a '-d' or '-s' parameter,the script will execute the tasks defined in task files with a "testsuite.json" suffix from the dir given by '-d' or a single task file given by '-s'.
An example task file is given in examples/test/demo/temp_testsuite.json including a training and a prediction task. <br>

3.Config files
------
default_env.json <br>
Please set role id in "role", including host, guest, and arbiter.<br>
Please build the relationship between roles and ip in "ip_map",where -1 stands for local,and remote host will be given ip address. <br>

testsuite.json <br>
You can submit data for many tasks once in "data",and each has a series of configs in a dict.<br>
"role" parameter describes the location of the data defined in default_env.json.For example, "guest_0" represents the data located in the first guest defined in the guest list of default_env.json. <br>
You can define your own tasks in "tasks".Training tasks and prediction tasks are supported now. There is some difference between them.<br>
A prediction task needs to state the task name of the training task which it depends on. <br>
Please name different tasks with different names,if two tasks share the same name,you will get the execution result of the letter defined. <br>

demo:<br>
```shell script
python run_test.py default_env.json -s ./demo/temp_testsuite.json
```
4.Examples of results
------

```text
./demo/temp_testsuite.json
====================================================================
lr	success	201912271619411350983
lr-predict	success	201912271620429623264
```


