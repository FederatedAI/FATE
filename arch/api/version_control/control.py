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
from arch.api.utils.core import get_scene_key, json_loads, json_dumps, bytes_to_string
from arch.api.utils import log_utils
import traceback
LOGGER = log_utils.getLogger()


def new_table_info_for_save(namespace=None, data_type=None, scene_id=None, my_role=None, my_party_id=None,
                            partner_party_id=None, tag=None, branch="master"):
    if not namespace:
        if data_type:
            namespace = get_scene_namespace(data_type=data_type, scene_id=scene_id, my_role=my_role,
                                            my_party_id=my_party_id, partner_party_id=partner_party_id)
        else:
            return None, None
    # Create new commit id for saving, branch current commit as parent
    # TODO: get job id from task manager
    commit_id = eggroll.get_job_id()
    name = commit_id  # commit id as name
    save_commit_tmp(commit_id=commit_id, data_table_namespace=namespace, tag=tag, branch=branch)
    return name, namespace


def get_table_info_for_read(namespace=None, data_type=None, scene_id=None, my_role=None, my_party_id=None,
                            partner_party_id=None, commit_id=None, tag=None, branch="master"):
    if not namespace:
        if data_type:
            namespace = get_scene_namespace(data_type=data_type, scene_id=scene_id, my_role=my_role,
                                            my_party_id=my_party_id, partner_party_id=partner_party_id)
        else:
            return None, None
    name = None
    if commit_id:
        name = commit_id
    else:
        version_table = get_version_table(data_table_namespace=namespace)
        branch_current_commit = get_branch_current_commit(version_table=version_table, branch_name=branch)
        if branch_current_commit:
            # Return branch current commit id for reading
            name = branch_current_commit
    return name, namespace


def save_version(name, namespace, version_log='', tag=None, branch=None):
    try:
        if not tag or not branch:
            tmp_tag, tmp_branch = get_commit_tmp(commit_id=name, data_table_namespace=namespace)
            tag = tmp_tag if not tag else tag
            branch = tmp_branch if not branch else branch
        save_version_info(commit_id=name, data_table_namespace=namespace, version_log=version_log, tag=tag, branch=branch)
        return True
    except Exception as e:
        LOGGER.exception(e)
        return False
    finally:
        delete_commit_tmp(commid_id=name, data_table_namespace=namespace)


def save_version_info(commit_id, data_table_namespace, version_log, tag, branch):
    version_table = get_version_table(data_table_namespace=data_table_namespace)
    parent = get_branch_current_commit(version_table=version_table, branch_name=branch)
    version_info = dict()
    version_info["commitId"] = commit_id
    if parent != commit_id:
        version_info["parent"] = parent
    else:
        version_info.update(get_version_info(version_table=version_table, commit_id=parent))
        version_info["repeatCommit"] = True
    version_info["name"] = commit_id
    version_info["namespace"] = data_table_namespace
    version_info["log"] = version_log
    version_info["tag"] = tag
    version_table.put(commit_id, json_dumps(version_info), use_serialize=False)
    # todo: should be use a lock
    version_table.put(branch, commit_id, use_serialize=False)


def version_history(namespace=None, data_type=None, scene_id=None, my_role=None, my_party_id=None,
                    partner_party_id=None, commit_id=None, tag=None, branch="master", limit=10):
    if not namespace:
        if data_type:
            namespace = get_scene_namespace(data_type=data_type, scene_id=scene_id, my_role=my_role,
                                            my_party_id=my_party_id, partner_party_id=partner_party_id)
        else:
            return None, None
    version_table = get_version_table(data_table_namespace=namespace)
    historys = list()
    if commit_id:
        # Get this commit information
        historys.append(get_version_info(version_table=version_table, commit_id=commit_id))
    else:
        branch_current_commit = get_branch_current_commit(version_table=version_table, branch_name=branch)
        if branch_current_commit:
            commit_id = branch_current_commit
            for i in range(limit):
                info = get_version_info(version_table=version_table, commit_id=commit_id)
                if info:
                    commit_info = json_loads(info)
                    historys.append(commit_info)
                    commit_id = commit_info["parent"]
                else:
                    break
    return historys


def get_scene_namespace(data_type, scene_id=None, my_role=None, my_party_id=None, partner_party_id=None):
    if not (scene_id and my_role and my_party_id and partner_party_id):
        scene_id, my_role, my_party_id, partner_party_id = get_default_scene_info()
    scene_key = get_scene_key(scene_id, my_role, my_party_id, partner_party_id)
    return "%s_%s" % (scene_key, data_type)


def get_version_table(data_table_namespace):
    version_table = eggroll.table(name=data_table_namespace, namespace="version_control",
                                  partition=1, create_if_missing=True, error_if_exist=False)
    return version_table


def get_branch_current_commit(version_table, branch_name):
    try:
        return bytes_to_string(version_table.get(branch_name, use_serialize=False))
    except:
        return None


def get_version_info(version_table, commit_id):
    info = version_table.get(commit_id, use_serialize=False)
    if info:
        return json_loads(info)
    else:
        return dict()


def get_default_scene_info():
    runtime_conf = federation.get_runtime_conf()
    scene_id = runtime_conf.get("local", {}).get("scene_id")
    my_role = runtime_conf.get("local", {}).get("role")
    my_party_id = runtime_conf.get("local", {}).get("party_id")
    partner_party_id = runtime_conf.get("role", {}).get("host" if my_role == "guest" else "guest")[0]
    return scene_id, my_role, my_party_id, partner_party_id


def get_commit_tmp_table(data_table_namespace):
    version_tmp_table = eggroll.table(name=data_table_namespace, namespace="version_tmp",
                                      partition=1, create_if_missing=True, error_if_exist=False)
    return version_tmp_table


def save_commit_tmp(commit_id, data_table_namespace, tag, branch):
    version_tmp_table = get_commit_tmp_table(data_table_namespace=data_table_namespace)
    version_tmp_table.put(commit_id, json_dumps({"tag": tag, "branch": branch}), use_serialize=False)


def get_commit_tmp(commit_id, data_table_namespace):
    version_tmp_table = get_commit_tmp_table(data_table_namespace=data_table_namespace)
    commit_tmp_info = version_tmp_table.get(commit_id, use_serialize=False)
    if commit_tmp_info:
        commit_tmp = json_loads(commit_tmp_info)
        return commit_tmp["tag"], commit_tmp["branch"]
    else:
        return None, "master"


def delete_commit_tmp(commid_id, data_table_namespace):
    version_tmp_table = get_commit_tmp_table(data_table_namespace=data_table_namespace)
    version_tmp_table.delete(commid_id)
