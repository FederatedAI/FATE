#!/bin/bash
set -e
cd /fate/python/fate_flow
nohup python fate_flow_server.py &

cd /fate/fateboard
${JAVA_PATH}/java -Dspring.config.location=/fate/fateboard/conf/application.properties -Dssh_config_file=/fate/fateboard/ssh/ -DFATE_DEPLOY_PREFIX=/fate/logs/ -Xmx2048m -Xms2048m -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:gc.log -XX:+HeapDumpOnOutOfMemoryError -jar fateboard.jar


exec "$@"
