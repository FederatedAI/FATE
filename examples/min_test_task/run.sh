role=${1}
task=${2}
if [ ${role} == "guest" ];then
    echo "role is guest"
    party='b'
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

cd ../../arch/task_manager/
python $basepath/run_task.py upload $role ${data} intersect
sleep 2
echo "finish upload intersect data"
python $basepath/run_task.py upload $role ${data} train
sleep 2
echo "finish upload train data"
python $basepath/run_task.py upload $role ${data} predict
sleep 2
echo "finish upload predict data"

if [ ${role} == "guest" ];then
    python $basepath/run_task.py all ${task}
fi

echo "*********************"
echo "*******finish!*******"
