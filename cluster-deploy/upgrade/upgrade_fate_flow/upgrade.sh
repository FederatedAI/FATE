#!/bin/bash


current_version=v1.2.0
fate_dir=/data/projects/fate


dir=$(dirname $(readlink -f "$0"))
python_dir="$dir/$current_version/python"

echo "start update file"
cp -rf $python_dir $fate_dir

# update settings
python modify_settings.py "$fate_dir/python/fate_flow/settings.py"
sed -i 's/\r$//' "$fate_dir/python/fate_flow/settings.py"
echo "update file success"  

if [[ $current_version == "v1.2.0" ]];then
    sed -i.bak "s/SERVINGS = CenterConfig.get_settings.*//g" "$fate_dir/python/fate_flow/settings.py"
    sed -i.bak "s/use_zk=USE_CONFIGURATION_CENTER, hosts=ZOOKEEPER_HOSTS.*//g" "$fate_dir/python/fate_flow/settings.py"
    use_zk=USE_CONFIGURATION_CENTER, hosts=ZOOKEEPER_HOSTS
fi

sh "$fate_dir/python/fate_flow/service.sh" restart




