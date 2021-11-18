project_base=$(cd `dirname $0`;pwd)
cd ${project_base}

new_repo_file="CentOS-Base.repo"

if [[ -f ${new_repo_file} ]];then
    echo "[INFO] start replace repo file"
    mv /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.backup
    cp ${new_repo_file} /etc/yum.repos.d/CentOS-Base.repo
    yum clean all
    #yum makecache
    sh ./bin/install_os_dependencies.sh root
    yum clean all
    cp /etc/yum.repos.d/CentOS-Base.repo.backup /etc/yum.repos.d/CentOS-Base.repo
else
    echo "[INFO] use default repo file"
    sh ./bin/install_os_dependencies.sh root
    yum clean all
fi