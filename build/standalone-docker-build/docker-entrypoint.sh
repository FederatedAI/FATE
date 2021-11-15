#!/bin/bash
set -e
cd ${WORKDIR}
cd ${WORKDIR}/fateflow/python/fate_flow
nohup python fate_flow_server.py &

cd ${WORKDIR}/fateboard
${JAVA_PATH}/java -Dspring.config.location=${WORKDIR}/fateboard/conf/application.properties -Dssh_config_file=${WORKDIR}/fateboard/ssh/ -DFATE_DEPLOY_PREFIX=${WORKDIR}/fateboard/logs/ -Xmx2048m -Xms2048m -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:gc.log -XX:+HeapDumpOnOutOfMemoryError -jar fateboard.jar


exec "$@"
