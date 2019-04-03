#!/bin/bash

user=app
dir=/data/projects/fate
mysqldir=/data/projects/common/mysql/mysql-8.0.13
javadir=/data/projects/common/jdk/jdk1.8.0_192
partylist=(10000 10001)
JDBC0=(172.16.153.63 eggroll_meta root Fate123#$)
JDBC1=(172.16.153.57 eggroll_meta root Fate123#$)
roleiplist=(172.16.153.63 172.16.153.63 172.16.153.71 172.16.153.121
172.16.153.57 172.16.153.57 172.16.153.97 172.16.153.88)
egglist0=(172.16.153.xx)
egglist1=(172.16.153.xx)
exchangeip=172.16.153.113
