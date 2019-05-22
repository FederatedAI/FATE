#!/bin/bash
mkdir -p /data/projects/common/jdk
mkdir -p /data/projects/fate
chown -R app:apps ./*
cd /data
chown -R app:apps projects
yum -y install gcc gcc-c++ make autoconfig openssl-devel supervisor gmp-devel mpfr-devel libmpc-devel libaio numactl autoconf automake libtool libffi-dev
