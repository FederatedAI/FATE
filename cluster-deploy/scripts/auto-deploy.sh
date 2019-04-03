#!/bin/bash

version=0.1
cwd=`pwd`

cd ../../
fate_dir=`pwd`
base_dir=$fate_dir/arch
output_dir=$fate_dir/cluster-deploy/example-dir-tree

cd $cwd
source ./configurations.sh

cd $output_dir

tar -czf fate.tar ./*
for ((i=0;i<${#roleiplist[*]};i++))
do
	eval ip=\${roleiplist[${i}]}
	scp fate.tar $user@$ip:$dir
	ssh -tt $user@$ip<< eeooff
	cd $dir
	tar -xzf fate.tar
	rm -f fate.tar
	exit
eeooff
done

cd $output_dir
for ((i=0;i<${#partylist[*]};i++))
do
	f=$((4*${i}+0))
	m=$((4*${i}+1))
	p=$((4*${i}+2))
	r=$((4*${i}+3))
	eval fip=\${roleiplist[${f}]}
	eval mip=\${roleiplist[${m}]}
	eval pip=\${roleiplist[${p}]}
	eval rip=\${roleiplist[${r}]}
	eval partyid=\${partylist[${i}]}
	eval jdbcip=\${JDBC${i}[0]}
	eval jdbcdbname=\${JDBC${i}[1]}
	eval jdbcuser=\${JDBC${i}[2]}
	eval jdbcpasswd=\${JDBC${i}[3]}
	eval l=\${#egglist${i}[*]}
	for ((j=0;j<$l;j++))
	do
		eval ip=\${egglist${i}[${j}]}
		scp fate.tar $user@$ip:$dir
		ssh -tt $user@$ip<< eeooff
		cd $dir
		tar -xzf fate.tar
		rm -f fate.tar
		sed -i "s/party.id=.*/party.id=$partyid/g" ./egg/conf/egg.properties
		exit
eeooff
	done

	if [ ! $exchangeip ]
	then
		if [ $(($i%2)) == 0 ]
		then
			j=$((4*($i+1)+2))
			eval exchangeip=\${roleiplist[${j}]}
		else
			j=$((4*($i-1)+2))
			eval exchangeip=\${roleiplist[${j}]}
		fi
	else
		scp fate.tar $user@$exchangeip:$dir
		echo exchangeip=$exchangeip
		ssh -tt $user@$exchangeip << eeooff
		cd $dir
		if [[ ! -d proxy ]]
		then
			tar -xzf fate.tar
			rm -f fate.tar
			sed -i "6s/:.*/: \"$exchangeip\",/g" ./proxy/conf/route_table.json
			if [ $i == 0 ]
			then
				sed -i "11s/\".*\"/\"$partyid\"/g" ./proxy/conf/route_table.json
				sed -i "12s/default/fate/g" ./proxy/conf/route_table.json
				sed -i "14s/:.*/: \"$fip\",/g" ./proxy/conf/route_table.json
			elif [ $i == 1 ]
			then
				sed -i "19s/\".*\"/\"$partyid\"/g" ./proxy/conf/route_table.json
				sed -i "20s/default/fate/g" ./proxy/conf/route_table.json
				sed -i "22s/:.*/: \"$fip\",/g" ./proxy/conf/route_table.json
			else
				echo "please add configuration of exchange role!"
			fi
		else
			rm -f fate.tar
			echo "please add configuration of exchange role!"
		exit
eeooff
	fi
	ssh -tt $user@$fip << eeooff
	cd $dir
	sed -i "s/party.id=.*/party.id=$partyid/g" ./federation/conf/federation.properties
	sed -i "s/meta.service.ip=.*/meta.service.ip=$mip/g" ./federation/conf/federation.properties
	sed -i "4s/:.*/: \"$rip\",/g" ./python/arch/conf/server_conf.json
	sed -i "8s/:.*/: \"$fip\",/g" ./python/arch/conf/server_conf.json
	sed -i "s/party.id=.*/party.id=$partyid/g" ./egg/conf/egg.properties
	exit
eeooff
	echo federation module of $partyid done!
	ssh -tt $user@$mip << eeooff
	cd $dir
	sed -i "s/party.id=.*/party.id=$partyid/g" ./meta-service/conf/meta-service.properties
	sed -i "s#//.*?#//$jdbcip:3306/$jdbcdbname?#g" ./meta-service/conf/jdbc.properties
	sed -i "s/jdbc.username=.*/jdbc.username=$jdbcuser/g" ./meta-service/conf/jdbc.properties
	sed -i "s/jdbc.password=.*/jdbc.password=$jdbcpasswd/g" ./meta-service/conf/jdbc.properties
	sed -i "4s/:.*/: \"$rip\",/g" ./python/arch/conf/server_conf.json
	sed -i "8s/:.*/: \"$fip\",/g" ./python/arch/conf/server_conf.json
	sed -i "s/party.id=.*/party.id=$partyid/g" ./egg/conf/egg.properties
	exit
eeooff
	echo meta-service module of $partyid done!
	ssh -tt $user@$pip << eeooff
	cd $dir
	sed -i "s/coordinator=.*/coordinator=$partyid/g" ./proxy/conf/proxy.properties
	sed -i "s/ip=.*/ip=$pip/g" ./proxy/conf/proxy.properties
	sed -i "6s/:.*/: \"$exchangeip\",/g" ./proxy/conf/route_table.json
	sed -i "11s/\".*\"/\"$partyid\"/g" ./proxy/conf/route_table.json
	sed -i "14s/:.*/: \"$fip\",/g" ./proxy/conf/route_table.json
	sed -i "4s/:.*/: \"$rip\",/g" ./python/arch/conf/server_conf.json
	sed -i "8s/:.*/: \"$fip\",/g" ./python/arch/conf/server_conf.json
	sed -i "s/party.id=.*/party.id=$partyid/g" ./egg/conf/egg.properties
	exit
eeooff
	echo proxy module of $partyid done!
	ssh -tt $user@$rip << eeooff
	cd $dir
	sed -i "s/party.id=.*/party.id=$partyid/g" ./roll/conf/roll.properties
	sed -i "s/meta.service.ip=.*/meta.service.ip=$mip/g" ./roll/conf/roll.properties
	sed -i "4s/:.*/: \"$rip\",/g" ./python/arch/conf/server_conf.json
	sed -i "8s/:.*/: \"$fip\",/g" ./python/arch/conf/server_conf.json
	sed -i "s/party.id=.*/party.id=$partyid/g" ./egg/conf/egg.properties
	exit
eeooff
	echo roll module of $partyid done!
	ssh -tt $user@$jdbcip<< eeooff
	${mysqldir}/bin/mysql -u$jdbcuser -p$jdbcpasswd -S ${mysqldir}/mysql.sock
	create database $jdbcdbname;
	use $jdbcdbname;
	create table node(
	ip varchar(20),
	port varchar(20),
	type varchar(20),
	status varchar(20),
	CONSTRAINT pk_person PRIMARY KEY (ip,port)
	);
	INSERT INTO node (ip, port, type, status) values ('${rip}', '8011', 'ROLL', 'HEALTHY');
	INSERT INTO node (ip, port, type, status) values ('${fip}', '7888', 'EGG', 'HEALTHY');
	INSERT INTO node (ip, port, type, status) values ('${mip}', '7888', 'EGG', 'HEALTHY');
	INSERT INTO node (ip, port, type, status) values ('${pip}', '7888', 'EGG', 'HEALTHY');
	INSERT INTO node (ip, port, type, status) values ('${rip}', '7888', 'EGG', 'HEALTHY');
	INSERT INTO node (ip, port, type, status) values ('${fip}', '7778', 'STORAGE', 'HEALTHY');
	INSERT INTO node (ip, port, type, status) values ('${mip}', '7778', 'STORAGE', 'HEALTHY');
	INSERT INTO node (ip, port, type, status) values ('${pip}', '7778', 'STORAGE', 'HEALTHY');
	INSERT INTO node (ip, port, type, status) values ('${rip}', '7778', 'STORAGE', 'HEALTHY');
	INSERT INTO node (ip, port, type, status) values ('${pip}', '9370', 'PROXY', 'HEALTHY');
	exit;
	exit
eeooff
done

cd $cwd