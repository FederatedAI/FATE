role=$1
job_id=$2
script=$3

echo "start to run compiler"
script_path=$(cd "$(dirname "$0")";pwd)
compiler_path=$(cd "$(dirname "$0")";pwd)'/compiler/'
source $compiler_path'antlr4.sh'
antlr4 $compiler_path'Fml.g4' -Dlanguage=Python3 -visitor -o $compiler_path'parser/'                                                                                                                                                                                      
python $compiler_path'fateScript2Py.py' $script_path'/'$script
echo "Finish compiling"


generator_path=$(cd "$(dirname "$0")";pwd)'/../../federatedml/util/transfer_variable_generator.py'
transfer_json_path=$(cd "$(dirname "$0")";pwd)'/conf/FateScriptTransferVar.json'
transfer_variable_path=$(cd "$(dirname "$0")";pwd)'/utils/fate_script_transfer_variable.py'
python $generator_path $transfer_json_path $transfer_variable_path
echo 'Finish generate fate_script_transfer_variable.py'

#0:standalone   1:cluster
work_mode=1

base_conf_path=$(cd "$(dirname "$0")";pwd)'/conf/'
runtime_conf_path=$base_conf_path$role"_runtime_conf.json"
echo "runtime_conf_path:"$runtime_conf_path

python fateScript.py $role $job_id $runtime_conf_path $work_mode&

sleep 1
