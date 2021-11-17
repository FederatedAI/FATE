#!/bin/bash
set -e
cd ${WORKDIR}

cd ${WORKDIR}/fateflow
sh bin/service.sh start

cd ${WORKDIR}/fateboard
sh service.sh starting

exec "$@"
