role=${1}
task=${2}

if [ ! ${role} ];then
    echo "Lack of parameter role, for host user, usage:"
    echo "sh run.sh host fast(or normal)"
    echo "Guest user usage: "
    echo "sh run.sh guest fast(or normal) host_table_name host_namespace"
    echo "where host_table_name & host_namespace are provided by host user"
    exit 0
fi

if [ ! ${task} ];then
    echo "Lack of parameter mode, for host user, usage: sh run.sh host fast(or normal)"
    echo "sh run.sh host fast(or normal)"
    echo "Guest user usage: "
    echo "sh run.sh guest fast(or normal) host_table_name host_namespace"
    echo "where host_table_name & host_namespace are provided by host user"
    exit 0
fi

if [ ${role} == "guest" ];then
    echo "role is guest"
    party='b'
    host_name=${3}
    host_namespace=${4}

    if [[ ! ${host_name} ]] || [[ ! ${host_namespace} ]];then
        echo "Guest user usage: "
        echo "sh run.sh guest fast(or normal) host_table_name host_namespace"
        echo "where host_table_name & host_namespace are provided by host user"
        exit 0
    fi


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
    python $basepath/run_task.py upload $role ${data}
    sleep 2
    echo "finish upload intersect data"
fi

if [ ${role} == "guest" ];then
    python $basepath/run_task.py all ${task} ${data} ${host_name} ${host_namespace}
fi

echo "*********************"
echo "*******finish!*******"
