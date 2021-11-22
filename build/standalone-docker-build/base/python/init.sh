project_base=$(cd `dirname $0`;pwd)
cd ${project_base}

new_repo_file="CentOS-Base.repo"

if [[ -f ${new_repo_file} ]];then
    echo "[INFO] use replace repo file"
    mv /etc/yum.repos.d /etc/yum.repos.d.bak
    mkdir -p /etc/yum.repos.d
    cp ${new_repo_file} /etc/yum.repos.d/CentOS-Base.repo
    echo "[INFO] replace repo file"
    echo "[INFO] new repo:"
    cat /etc/yum.repos.d/CentOS-Base.repo
    #yum makecache
    #echo "[INFO] macke cache"
    sh ./install_os_dependencies.sh root
    yum clean all
    rm -rf /etc/yum.repos.d
    mv /etc/yum.repos.d.bak /etc/yum.repos.d
    echo "[INFO] restore repo file"
else
    echo "[INFO] use default repo file"
    sh ./install_os_dependencies.sh root
    yum clean all
fi
