#!/bin/bash

version=v1.3.2
upload_file=(fate_flow/apps/data_access_app.py fate_flow/driver/job_controller.py fate_flow/driver/task_scheduler.py federatedml/tree/hetero_decision_tree_guest.py fate_flow/fate_flow_client.py)

mkdir -vp ${version}/python

for file in ${upload_file[@]}
do
    dir=`dirname $file`
    mkdir -vp ${version}/python/$dir
    cp ../../$file ${version}/python/$file
done

sed -i.bak "s/current_version=.*/current_version=${version}/g" upgrade.sh
cp upgrade.sh ./$version
cp modify_settings.py ./$version

zip -r ${version}.zip $version




