#!/bin/bash
set -euo pipefail

################################  请根据实际情况修改以下参数  ################################
FATE_ROOT=/data/projects/fate
DB_USER=root
DB_PASS=fate_dev
MYSQL_ROOT=$FATE_ROOT/common/mysql/mysql-8.0.13
MYSQL_SOCKET_PATH=$MYSQL_ROOT/run/mysql.sock

# Choose one from 'allinone' and 'ansible'
DEPLOY_METHOD=ansible
# If the deploy method is 'ansible', please make sure where the supervisord files are.
SUPERVISOR_ROOT=/data/projects/common/supervisord
################################  请根据实际情况修改以上参数  ################################

FATE_PYTHON_ROOT=$FATE_ROOT/python
FATEBOARD_ROOT=$FATE_ROOT/fateboard

SCRIPT_PATH=$(dirname $(readlink -f "$0"))
PYTHON_PACKAGE_PATH=$SCRIPT_PATH/python.tar.gz

export FATE_ROOT=$FATE_ROOT
export FATE_PYTHON_ROOT=$FATE_PYTHON_ROOT
export FATEBOARD_ROOT=$FATEBOARD_ROOT
export SCRIPT_PATH=$SCRIPT_PATH
export PYTHON_PACKAGE_PATH=$PYTHON_PACKAGE_PATH
export DB_USER=$DB_USER
export DB_PASS=$DB_PASS
export DEPLOY_METHOD=$DEPLOY_METHOD
export MYSQL_ROOT=$MYSQL_ROOT
export MYSQL_SOCKET_PATH=$MYSQL_SOCKET_PATH
export CURDATE=`date -d today +"%Y%m%d%H%M%S"`


function check_all() {
  check_deploy_method
  check_fate_root
  check_mysql_conn
  check_board_root
}


function check_deploy_method() {
  echo "INFO: Checking progress starting..."

  echo "INFO: Checking if deploy method is valid..."
  if [ "$DEPLOY_METHOD" != "ansible" ] && [ "$DEPLOY_METHOD" != "allinone" ]; then
    echo "ERROR: please choose deploy method from one of 'allinone' and 'ansible'. Upgrade process aborting.";
    exit 1;
  else
    echo "INFO: Deploy method is $DEPLOY_METHOD";
  fi

  case $DEPLOY_METHOD in
    "ansible")
      if [ ! -f $SUPERVISOR_ROOT/service.sh ]; then
        echo "ERROR: SUPERVISOR execute file $SUPERVISOR_ROOT/service.sh not exists. Upgrade process aborting.";
        exit 1;
      fi;;
  esac
}


function check_fate_root() {
  echo "INFO: Checking if fate root $FATE_ROOT exists..."
  if [ ! -d $FATE_ROOT ]; then
    echo "ERROR: Fate root $FATE_ROOT does not exist. Upgrade process aborting.";
    exit 1;
  else
    echo "INFO: Done."
  fi


  echo "INFO: Checking if python package path $PYTHON_PACKAGE_PATH exists..."
  if [ ! -f $PYTHON_PACKAGE_PATH ]; then
    echo "ERROR: New python package $PYTHON_PACKAGE_PATH does not exist. Upgrade process aborting.";
    exit 1;
  else
    echo "INFO: Done."
  fi
}


function check_mysql_conn() {
  echo "INFO: Checking if mysql root $MYSQL_ROOT exists..."
  if [ ! -d $MYSQL_ROOT ]; then
    echo "ERROR: Mysql root $FATE_ROOT does not exist. Upgrade process aborting.";
    exit 1;
  else
    echo "INFO: Done."
  fi
  
  $MYSQL_ROOT/bin/mysql -u$DB_USER -p$DB_PASS -S $MYSQL_SOCKET_PATH -e "select 'x';"
  if [ $? -ne 0 ]; then
    echo "ERROR: Mysql connect failed. Upgrade process aborting.";
    exit 1;
  else
    echo "INFO: Done."
  fi

  echo "INFO: Checking progress finished."
}


function check_board_root() {
  echo "INFO: Checking if fatetboard root $FATEBOARD_ROOT exists..."
  if [ ! -d $FATEBOARD_ROOT ]; then
    echo "ERROR: Fateboard root $FATEBOARD_ROOT does not exist. Upgrade process aborting.";
    exit 1;
  fi
}


function stop_fateflow() {
  echo "INFO: Stopping fate flow server..."
  case $DEPLOY_METHOD in
    "allinone")
      sh $FATE_PYTHON_ROOT/fate_flow/service.sh stop
      ;;
    "ansible")
      sh $SUPERVISOR_ROOT/service.sh stop fate-fateflow
      ;;
  esac
}


function start_fateflow() {
  echo "INFO: Starting fate flow server..."
  case $DEPLOY_METHOD in
    "allinone")
      sh $FATE_PYTHON_ROOT/fate_flow/service.sh start
      ;;
    "ansible")
      sh $SUPERVISOR_ROOT/service.sh start fate-fateflow
      ;;
  esac
}


function stop_fateboard() {
  echo "INFO: Stopping fateboard..."
  case $DEPLOY_METHOD in
    "allinone")
      sh $FATEBOARD_ROOT/service.sh stop
      ;;
    "ansible")
      sh $SUPERVISOR_ROOT/service.sh stop fate-fateboard
      ;;
  esac
}


function start_fateboard() {
  echo "INFO: Starting fateboard..."
  case $DEPLOY_METHOD in
    "allinone")
      sh $FATEBOARD_ROOT/service.sh start
      ;;
    "ansible")
      sh $SUPERVISOR_ROOT/service.sh update fate-fateboard
      sh $SUPERVISOR_ROOT/service.sh start fate-fateboard
      ;;
  esac
}


function backup_python() {
  echo "INFO: Backup for old python directory"
  PYTHON_BACKUP_PATH=$FATE_ROOT/python_150bak_$CURDATE
  export PYTHON_BACKUP_PATH=$PYTHON_BACKUP_PATH
  mv $FATE_PYTHON_ROOT $PYTHON_BACKUP_PATH
  echo "INFO: Backup successfully. Old python directory currently located at $PYTHON_BACKUP_PATH"
}


function copy_related_files() {
  echo "INFO: Copy related new code in version 1.5.0 into system"

  if [ -f $FATE_ROOT/fate.env ]; then
    mv $FATE_ROOT/fate.env $FATE_ROOT/fate.env_150bak_$CURDATE
    echo "$FATE_ROOT/fate.env has been renamed to $FATE_ROOT/fate.env_150bak_$CURDATE"
  fi

  cp $SCRIPT_PATH/other_codes/fate.env $FATE_ROOT
  echo "INFO: Done"
}


function replace_new_python() {
  echo "INFO: Unpacking new python package..."
  tar zxf $PYTHON_PACKAGE_PATH -C $FATE_ROOT
  echo "INFO: Python package unpacked successfully."
}


function execute_sql_upgrade() {
  echo "INFO: Upgrading database..."
  $MYSQL_ROOT/bin/mysql -u$DB_USER -p$DB_PASS -S $MYSQL_SOCKET_PATH < $SCRIPT_PATH/upgrade_sql/upgrade_fate_flow_db.sql
  echo "INFO: Upgrade database finished."
}


function install_dependency() {
  echo "INFO: Installing python dependency..."
  source $FATE_ROOT/bin/init_env.sh
  pip install -r $SCRIPT_PATH/requirements.txt --no-index --find-links=$SCRIPT_PATH/fate_dependencies
  echo "INFO: Dependency has been installed successfully."
}


function backup_fateboard() {
  echo "INFO: Backup for old fateboard directory"
  FATEBOARD_BACKUP_PATH=$FATE_ROOT/fateboard_150bak_$CURDATE
  cp -r $FATEBOARD_ROOT $FATEBOARD_BACKUP_PATH
  echo "INFO: Backup successfully. Old fateboard directory currently located at $FATEBOARD_BACKUP_PATH"
}


function replace_fateboard() {
  echo "INFO: Replacing new fateboard..."
  if [ -f $FATEBOARD_ROOT/fateboard-1.5.0.jar ] ; then
    mv $FATEBOARD_ROOT/fateboard-1.5.0.jar $FATEBOARD_ROOT/fateboard-1.5.0.jar_150bak_$CURDATE
  fi
  if [ -L $FATEBOARD_ROOT/fateboard.jar ] ; then
    rm $FATEBOARD_ROOT/fateboard.jar
  fi
  if [ -f $FATEBOARD_ROOT/service.sh ]; then
    mv $FATEBOARD_ROOT/service.sh $FATEBOARD_ROOT/service.sh_150bak_$CURDATE
  fi

  cp $SCRIPT_PATH/fateboard/fateboard-1.5.1.jar $FATEBOARD_ROOT
  cp $SCRIPT_PATH/fateboard/service.sh $FATEBOARD_ROOT
  ln -s $FATEBOARD_ROOT/fateboard-1.5.1.jar $FATEBOARD_ROOT/fateboard.jar
  echo "INFO: Done."
}


function upgrade_all() {
  echo "INFO: Upgrading process start..."
  check_all
  stop_fateflow
  backup_python
  replace_new_python
  copy_related_files
  install_dependency
  execute_sql_upgrade
  start_fateflow
  backup_fateboard
  replace_fateboard
  stop_fateboard
  start_fateboard
  echo "INFO: Upgrading process finished."
}


function upgrade_mysql() {
  echo "INFO: Upgrading mysql process start..."
  # check
  check_mysql_conn
  # upgrade
  execute_sql_upgrade
  echo "INFO: Upgrading mysql process finished."
}


function upgrade_fate_python() {
  echo "INFO: Upgrading fate python process start..."
  check_deploy_method
  check_fate_root
  stop_fateflow
  backup_python
  replace_new_python
  install_dependency
  start_fateflow
  echo "INFO: Upgrading fate python process finished."
}



function upgrade_fateboard() {
  echo "INFO: Upgrading fateboard process start..."
  check_board_root
  backup_fateboard
  replace_fateboard
  stop_fateboard
  start_fateboard
  echo "INFO: Upgrading fateboard process finished."
}


case "$1" in
    fatepython)
        upgrade_fate_python
        ;;
    mysql)
        upgrade_mysql
        ;;
    fateboard)
        upgrade_fateboard
        ;;
    all)
        upgrade_all
        ;;
    *)
        echo "usage: $0 {fatepython|mysql|fateboard|all}"
        exit -1
esac
