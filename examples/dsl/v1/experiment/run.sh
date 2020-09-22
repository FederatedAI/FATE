role=${1}
work_mode=${2}


if [[ ! ${role} ]] || [[ ! ${work_mode} ]];then
    echo "Lack of parameter role, for host user, usage:"
    echo "sh run.sh host work_mode"
    echo "Guest user usage: "
    echo "sh run.sh guest work_mode guest_id host_id arbiter_id"
    exit 0
fi



if [ ${role} == "guest" ];then
    echo "role is guest"
    guest_id=${3}
    host_id=${4}
    arbiter_id=${5}

    if [[ ! ${guest_id} ]] || [[ ! ${host_id} ]] || [[ ! ${arbiter_id} ]];then
        echo "Guest user usage: "
        echo "sh run.sh guest work_mode guest_id host_id arbiter_id"
        exit 0
    fi

elif [ ${role} == "host" ];then
    echo "role is host"
else
    echo "Not support "${role}", please user 'guest' or 'host'"
    exit
fi

basepath=$(cd `dirname $0`; pwd)

if [ ${role} == "host" ];then
    python $basepath/run_task.py upload ${role} ${work_mode}
    sleep 2
    echo "finish upload intersect data"
fi

if [ ${role} == "guest" ];then
    python $basepath/run_task.py all ${role} ${work_mode} ${guest_id} ${host_id} ${arbiter_id}
fi

echo "*********************"
echo "*******finish!*******"
