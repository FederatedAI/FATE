#!/bin/bash
groupadd -g 6000 apps
useradd -s /bin/sh -g apps -d /home/app app
passwd app
mkdir -p /data/projects/common/jdk
mkdir -p /data/projects/fate
cd /data
chown -R app:apps projects
yum -y install gcc gcc-c++ make autoconfig openssl-devel supervisor gmp-devel mpfr-devel libmpc-devel libaio numactl autoconf automake libtool libffi-dev

