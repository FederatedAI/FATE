DIRNAME=$0
if [ "${DIRNAME:0:1}" = "/" ];then
    CURDIR=`dirname $DIRNAME`
else
    CURDIR="`pwd`"/"`dirname $DIRNAME`"
fi
echo $CURDIR


generator_path=$(cd "$(dirname "$0")";pwd)'/../../federatedml/util/transfer_variable_generator.py'
transfer_json_path=$(cd "$(dirname "$0")";pwd)'/conf/FateScriptTransferVar.json'
transfer_variable_path=$(cd "$(dirname "$0")";pwd)'/fate_script_transfer_variable.py'
python $generator_path $transfer_json_path $transfer_variable_path
echo 'Finish generate fate_script_transfer_variable.py'

party_1='H'
party_2='G'
party_3='A'

curtime=$(date +%Y%m%d%H%M%S)
jobid=("hetero_logistic_regression_example_standalone_"${curtime})
base_conf_path=$(cd "$(dirname "$0")";pwd)'/conf/'




runtime_conf_path_1=$base_conf_path"host_runtime_conf.json"
runtime_conf_path_2=$base_conf_path"guest_runtime_conf.json"
runtime_conf_path_3=$base_conf_path"arbiter_runtime_conf.json"

echo $party_1" runtime conf path:"$runtime_conf_path_1
echo $party_2" runtime conf path:"$runtime_conf_path_2
echo $party_3" runtime conf path:"$runtime_conf_path_3

cp $runtime_conf_path_1 $runtime_conf_path_1"_"$jobid
cp $runtime_conf_path_2 $runtime_conf_path_2"_"$jobid
cp $runtime_conf_path_3 $runtime_conf_path_3"_"$jobid


echo 'Generate jobid:'$jobid
echo "Start run fateScript.py for "$party_1
python fateScript.py $party_1 $jobid $runtime_conf_path_1"_"$jobid& 
#python fateScript.py $party_1 $jobid "/data/projects/qijun/fate/python/examples/hetero_logistic_regression/conf/host_runtime_conf.json_hetero_logistic_regression_example_standalone_20190508194322"&

echo "Start run fateScript.py for "$party_2
python fateScript.py $party_2 $jobid $runtime_conf_path_2"_"$jobid&
#python fateScript.py $party_2 $jobid "/data/projects/qijun/fate/python/examples/hetero_logistic_regression/conf/guest_runtime_conf.json_hetero_logistic_regression_example_standalone_20190508194322"&

sleep 1
echo "Start run fateScript.py for "$party_3
python fateScript.py $party_3 $jobid $runtime_conf_path_3"_"$jobid
#python fateScript.py $party_3 $jobid "/data/projects/qijun/fate/python/examples/hetero_logistic_regression/conf/arbiter_runtime_conf.json_hetero_logistic_regression_example_standalone_20190508194322"

sleep 1
