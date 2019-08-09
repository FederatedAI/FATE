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