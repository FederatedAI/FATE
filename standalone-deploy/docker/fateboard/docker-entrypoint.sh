#!/bin/bash
set -e
cd /fate/fateboard
java -Dspring.config.location=conf/application.properties -Dssh_config_file=conf/ -Xmx2048m -Xms2048m -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:gc.log -XX:+HeapDumpOnOutOfMemoryError -jar fateboard.jar
exec "$@"
