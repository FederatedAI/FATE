script=$1


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

party_1='H'
party_2='G'
party_3='A'
#0:standalone   1:cluster
work_mode=0

curtime=$(date +%Y%m%d%H%M%S)
jobid=("hetero_logistic_regression_example_standalone_"${curtime})


echo 'Generate jobid:'$jobid
python run_fateScript_standalone.py $jobid $work_mode&
sleep 1
