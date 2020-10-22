#!/bin/bash
set -e
cd /fate/fateboard
java -Dspring.config.location=/fate/fateboard/conf/application.properties -Dssh_config_file=/fate/fateboard/ssh/ -DFATE_DEPLOY_PREFIX=/fate/logs/ -Xmx2048m -Xms2048m -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:gc.log -XX:+HeapDumpOnOutOfMemoryError -jar fateboard.jar

exec "$@"