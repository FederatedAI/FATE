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

In guest, make sure Host is finish
>sh run.sh guest fast

After a short period of waiting time, you can see the test case is successfully or not.
If mode fast is successful, run mode normal is necessary.

In Host, you should do this before guest
>sh run.sh host normal

In guest, make sure Host is finish
>sh run.sh guest normal