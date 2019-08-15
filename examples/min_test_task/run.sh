role=${1}
task=${2}
if [ ${role} == "guest" ];then
    echo "role is guest"
    party='b'
    host_name=${3}
    host_namespace=${4}
elif [ ${role} == "host" ];then
    echo "role is host"
    party='a'
else
    echo "Not support "${role}", please user 'guest' or 'host'"
    exit
fi

basepath=$(cd `dirname $0`; pwd)
if [ ${task} == "fast" ];then
    echo "task is fast"
    data=${basepath}/../data/breast_${party}.csv
elif [ ${task} == "normal" ];then
    echo "task is normal"
    data=${basepath}/../data/default_credit_${party}.csv
else
    echo "Not support "${task}", please user 'fast' or 'normal'"
fi

if [ ${role} == "host" ];then
    cd ../../arch/task_manager/
    python $basepath/run_task.py upload $role ${data}
    sleep 2
    echo "finish upload intersect data"
fi

if [ ${role} == "guest" ];then
    python $basepath/run_task.py all ${task} ${data} ${host_name} ${host_namespace}
fi

echo "*********************"
echo "*******finish!*******"
