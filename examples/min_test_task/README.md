### Description
This is the minimum test task for installation. Using this you can test if the installation of FATE is successful or not.

#### Test task case
It includes three cases:
1. Test data upload
2. Test intersection
3. Test algorithm, e.g. in the case is Hetero-lr
It includes two mode: fast and normal. For fast, you can use this to make sure if some wrong with installation quickly. After test fast mode, you should run normal mode as well.

#### Usage
copy this to installation/example/
>  cp -r min_test_task install_path/example/

In Host, you should do this before guest
>sh run.sh host fast

After running this command, table_name and namespace of uploaded data will be shown. Use them as input parameter of next step.

In guest, make sure Host is finish
>sh run.sh guest fast ${host_table_name} ${host_namespace}

After a short period of waiting time, you can see the test case is successfully or not.
If mode fast is successful, run mode normal is necessary.

Similar with fast mode, running the following two steps is enough.

In Host, you should do this before guest
>sh run.sh host normal

Get host table name and namespace

In guest, make sure Host is finish
>sh run.sh guest normal ${host_table_name} ${host_namespace}

#### Test Result

The process of min-test can be described as the following steps.

##### Host Party
>sh run.sh host normal(or fast)

In host part, uploading data is the only operation when calling the command.

##### Guest Party
>sh run.sh guest normal(or fast) ${host_table_name} ${host_namespace}

In guest party, there are three tests are going to be verified.

1. Upload Data test
    This is same with host part. Upload the data in Eggroll and check if DTable count match the number of your uploaded file.

2. Intersect
    Guest will start an intersect task which the expected intersect count is already known. After finish the intersect job, the intersected data will be download and check if this data length equal to expected count.

3. Hetero-lr Train
    After that, a hetero-lr modeling task will be started. This min-test scrip will keep checking the status of this task. As long as the task is finished, it obtain the evaluation result of this task and see if the auc match the expected auc for corresponding pre-defined data set.