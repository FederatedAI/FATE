#!/bin/bash

user=app
deploy_dir=/data/projects/fate
party_list=(10000 9999)
party_names=(a b)
db_auth=(root fate_dev)
redis_password=fate_dev
cxx_compile_flag=false

# services for a
a_mysql=
a_redis=
a_fate_flow=
a_fateboard=
a_federation=
a_proxy=

a_roll=
a_metaservice=
a_egg=()

# services for b
b_mysql=
b_redis=
b_fate_flow=
b_fateboard=
b_federation=
b_proxy=

b_roll=
b_metaservice=
b_egg=()
