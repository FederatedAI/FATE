#!/bin/bash

user=app
deploy_dir=/data/projects/fate
party_list=(10000 9999)
party_names=(a b)
db_auth=(root fate_dev)
redis_password=fate_dev
cxx_compile_flag=false

# services for a
a_mysql=10.211.55.6
a_redis=10.211.55.6
a_fate_flow=10.211.55.6
a_fateboard=10.211.55.6
a_federation=10.211.55.9
a_proxy=10.211.55.6

a_roll=10.211.55.6
a_metaservice=10.211.55.6
a_egg=(10.211.55.9)
a_storage_service=(10.211.55.6)

# services for b
b_mysql=10.211.55.9
b_redis=10.211.55.9
b_fate_flow=10.211.55.9
b_fateboard=10.211.55.9
b_federation=10.211.55.6
b_proxy=10.211.55.9

b_roll=10.211.55.6
b_metaservice=10.211.55.6
b_egg=(10.211.55.6 10.211.55.10)
b_storage_service=(10.211.55.9)
