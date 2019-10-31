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
from arch.api import session
from arch.api.utils.core import json_loads, json_dumps, bytes_to_string
from arch.api.utils import log_utils
LOGGER = log_utils.getLogger()


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
        delete_commit_tmp(commit_id=name, data_table_namespace=namespace)


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


def get_latest_commit(data_table_namespace, branch="master"):
    version_table = get_version_table(data_table_namespace=data_table_namespace)
    return get_branch_current_commit(version_table=version_table, branch_name=branch)


def version_history(data_table_namespace, commit_id=None, branch="master", limit=10):
    version_table = get_version_table(data_table_namespace=data_table_namespace)
    history = list()
    if commit_id:
        # Get this commit information
        history.append(get_version_info(version_table=version_table, commit_id=commit_id))
    else:
        branch_current_commit = get_branch_current_commit(version_table=version_table, branch_name=branch)
        if branch_current_commit:
            commit_id = branch_current_commit
            for i in range(limit):
                if not commit_id:
                    continue
                commit_info = get_version_info(version_table=version_table, commit_id=commit_id)
                if commit_info:
                    history.append(commit_info)
                    commit_id = commit_info["parent"]
                else:
                    break
    return history


def get_version_table(data_table_namespace):
    version_table = session.table(name=data_table_namespace, namespace="version_control",
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


def delete_commit_tmp(commit_id, data_table_namespace):
    version_tmp_table = get_commit_tmp_table(data_table_namespace=data_table_namespace)
    version_tmp_table.delete(commit_id)


def get_commit_tmp_table(data_table_namespace):
    version_tmp_table = session.table(name=data_table_namespace, namespace="version_tmp",
                                      partition=1, create_if_missing=True, error_if_exist=False)
    return version_tmp_table

