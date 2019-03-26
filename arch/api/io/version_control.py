#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from arch.api import federation
from arch.api import eggroll
from arch.api.utils.core import get_scene_key, json_loads, json_dumps, bytes_string
from arch.api import RuntimeInstance
from arch.api import WorkMode


def save_version(name_space, scene_id=None, my_party_id=None, partner_party_id=None, my_role=None, commit_id=None, tag=None, branch="master"):
    version_table, scene_key = get_version_table(name_space=name_space, scene_id=scene_id, my_party_id=my_party_id, partner_party_id=partner_party_id, my_role=my_role)
    data_table_info = dict()
    branch_current_commit = version_table.get(branch, use_serialize=False)
    if branch_current_commit:
        parent = bytes_string(branch_current_commit)
    else:
        parent = "0"
    if not commit_id:
        # Create new commit id for saving, branch current commit as parent
        # TODO: get job id from task manager
        commit_id = eggroll.get_field("job_id")
    print(scene_key, parent, commit_id)
    return version_table, data_table_info, scene_key, parent, commit_id


def read_version(name_space, scene_id=None, my_party_id=None, partner_party_id=None, my_role=None, commit_id=None, tag=None, branch="master"):
    version_table, scene_key = get_version_table(name_space=name_space, scene_id=scene_id, my_party_id=my_party_id, partner_party_id=partner_party_id, my_role=my_role)
    parent = None
    if commit_id:
        # Get this commit information
        data_table_info = get_version_info(version_table=version_table, commit_id=commit_id)
    else:
        data_table_info = dict()
        branch_current_commit = version_table.get(branch, use_serialize=False)
        if branch_current_commit:
            # Return branch current commit id for reading
            commit_id = bytes_string(branch_current_commit)
    return version_table, data_table_info, scene_key, parent, commit_id


def version_history(name_space, scene_id=None, my_party_id=None, partner_party_id=None, my_role=None, commit_id=None, tag=None, branch="master", limit=10):
    version_table, scene_key = get_version_table(name_space=name_space, scene_id=scene_id, my_party_id=my_party_id, partner_party_id=partner_party_id, my_role=my_role)
    data_table_infos = list()
    if commit_id:
        # Get this commit information
        data_table_infos.append(json_loads(version_table.get(commit_id, use_serialize=False)))
    else:
        branch_current_commit = version_table.get(branch, use_serialize=False)
        if branch_current_commit:
            commit_id = bytes_string(branch_current_commit)
            for i in range(10):
                info = version_table.get(commit_id, use_serialize=False)
                if info:
                    commit_info = json_loads(info)
                    data_table_infos.append(commit_info)
                    commit_id = commit_info["parent"]
                else:
                    break
    print(data_table_infos)
    return data_table_infos


def get_version_table(name_space, scene_id=None, my_party_id=None, partner_party_id=None, my_role=None):
    if not (scene_id and my_party_id and partner_party_id and my_role):
        scene_id, my_party_id, partner_party_id, my_role = get_default_scene_info()
    scene_key = get_scene_key(scene_id, my_party_id, partner_party_id, my_role)
    version_table_name_space, version_table_name, version_table_partition = name_space, scene_key, 1
    version_table = eggroll.table(version_table_name, version_table_name_space, partition=version_table_partition, create_if_missing=True, error_if_exist=False)
    return version_table, scene_key


def get_version_info(version_table, commit_id):
    info = version_table.get(commit_id, use_serialize=False)
    if info:
        return json_loads(info)
    else:
        return dict()


def get_default_scene_info():
    runtime_conf = federation.get_field("runtime_conf")
    scene_id = runtime_conf.get("local", {}).get("scene_id")
    my_role = runtime_conf.get("local", {}).get("role")
    my_party_id = runtime_conf.get("local", {}).get("party_id")
    partner_party_id = runtime_conf.get("role", {}).get("host" if my_role == "guest" else "guest")[0]
    return scene_id, my_party_id, partner_party_id, my_role


def gen_data_table_info(name_space, scene_key, commit_id):
    data_table_info = dict()
    data_table_info["tableNameSpace"], data_table_info["tableName"] = "%s_%s" % (scene_key, name_space), commit_id
    if name_space == 'model_data':
        # todo: max size limit?
        data_table_info["tablePartition"] = 4 if RuntimeInstance.MODE == WorkMode.CLUSTER else 1
    elif name_space == 'feature_data':
        data_table_info["tablePartition"] = 10 if RuntimeInstance.MODE == WorkMode.CLUSTER else 1
    elif name_space == 'feature_meta':
        data_table_info["tablePartition"] = 1 if RuntimeInstance.MODE == WorkMode.CLUSTER else 1
    elif name_space == 'feature_header':
        data_table_info["tablePartition"] = 1 if RuntimeInstance.MODE == WorkMode.CLUSTER else 1
    else:
        data_table_info["tablePartition"] = 1
    return data_table_info


def get_data_table(data_table_info, create_if_missing=True):
    return eggroll.table(data_table_info["tableName"], data_table_info["tableNameSpace"], partition=data_table_info["tablePartition"], create_if_missing=create_if_missing, error_if_exist=False)


def save_version_info(version_table, branch, commit_id, parent, data_table_info):
    version_info = dict()
    version_info["commitId"] = commit_id
    if parent != commit_id:
        version_info["parent"] = parent
    else:
        version_info.update(get_version_info(version_table=version_table, commit_id=parent))
        version_info["repeatCommit"] = True
    version_info.update(data_table_info)
    version_table.put(commit_id, json_dumps(version_info), use_serialize=False)
    # todo: should be use a lock
    version_table.put(branch, commit_id, use_serialize=False)
