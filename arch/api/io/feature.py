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
from arch.api.io.version_control import read_version, save_version, gen_data_table_info, get_data_table, save_version_info
from arch.api.utils.core import json_dumps, json_loads, bytes_string


def save_feature_header(features, labels, commit_log="", commit_id=None,  tag=None, branch="master"):
    version_table, data_table_info, scene_key, parent, commit_id = save_version("feature_header_version",
                                                                                 commit_id=commit_id,
                                                                                 tag=tag,
                                                                                 branch=branch)
    data_table_info = gen_data_table_info("feature_header", scene_key=scene_key, commit_id=commit_id)
    data_table = get_data_table(data_table_info=data_table_info, create_if_missing=True)
    data_table.put("features", json_dumps(features), use_pickle=False)
    data_table.put("labels", json_dumps(labels), use_pickle=False)

    # save version info
    data_table_info["commitLog"] = commit_log
    data_table_info["parent"] = parent
    save_version_info(version_table=version_table, branch=branch, commit_id=commit_id, data_table_info=data_table_info)
    return commit_id


def read_feature_header(commit_id=None, tag=None, branch="master"):
    version_table, data_table_info, scene_key, parent, commit_id = read_version("feature_header_version",
                                                                                 commit_id=commit_id,
                                                                                 tag=tag,
                                                                                 branch=branch)
    if commit_id:
        # Maybe param commit id or get commit id by current branch commit
        data_table_info = data_table_info if data_table_info else gen_data_table_info("feature_header", scene_key=scene_key, commit_id=commit_id)
        data_table = get_data_table(data_table_info=data_table_info, create_if_missing=False)
        return json_loads(data_table.get("features", use_pickle=False)), json_loads(data_table.get("labels", use_pickle=False))
    else:
        return None, None


def save_feature_meta(kv_meta, commit_log="", commit_id=None, tag="", branch="master"):
    version_table, data_table_info, scene_key, parent, commit_id = save_version("feature_version",
                                                                                 commit_id=commit_id,
                                                                                 tag=tag,
                                                                                 branch=branch)
    data_table_info = gen_data_table_info("feature_meta", scene_key=scene_key, commit_id=commit_id)
    data_table = get_data_table(data_table_info=data_table_info, create_if_missing=True)
    for k, v in kv_meta.items():
        data_table.put(k, json_dumps(v), use_pickle=False)

    # save version info
    data_table_info["commitLog"] = commit_log
    data_table_info["parent"] = parent
    save_version_info(version_table=version_table, branch=branch, commit_id=commit_id, data_table_info=data_table_info)
    return commit_id


def read_feature_meta(commit_id=None, tag=None, branch="master"):
    version_table, data_table_info, scene_key, parent, commit_id = read_version("feature_version",
                                                                                 commit_id=commit_id,
                                                                                 tag=tag,
                                                                                 branch=branch)
    feature_meta = dict()
    if commit_id:
        # Maybe param commit id or get commit id by current branch commit
        data_table_info = data_table_info if data_table_info else gen_data_table_info("feature_meta",
                                                                                      scene_key=scene_key,
                                                                                      commit_id=commit_id)
        data_table = get_data_table(data_table_info=data_table_info, create_if_missing=False)
        for k, v in data_table.collect(use_pickle=False):
            feature_meta[k] = bytes_string(v)
        return feature_meta
    else:
        return feature_meta


def save_feature_data(kv_data, scene_id, my_party_id, partner_party_id, my_role, commit_log="", commit_id=None, tag="", branch="master"):
    version_table, data_table_info, scene_key, parent, commit_id = save_version("feature_version",
                                                                                 scene_id=scene_id,
                                                                                 my_party_id=my_party_id,
                                                                                 partner_party_id=partner_party_id,
                                                                                 my_role=my_role,
                                                                                 commit_id=commit_id,
                                                                                 tag=tag,
                                                                                 branch=branch)
    data_table_info = gen_data_table_info("feature_data", scene_key=scene_key, commit_id=commit_id)
    data_table = get_data_table(data_table_info=data_table_info, create_if_missing=True)
    data_table.put_all(kv_data)
    # save version info
    data_table_info["commitLog"] = commit_log
    data_table_info["parent"] = parent
    save_version_info(version_table=version_table, branch=branch, commit_id=commit_id, data_table_info=data_table_info)
    return commit_id


def get_feature_data_table(scene_id=None, my_party_id=None, partner_party_id=None, my_role=None, commit_id=None, tag=None, branch="master"):
    version_table, data_table_info, scene_key, parent, commit_id = read_version("feature_version",
                                                                                 scene_id=scene_id,
                                                                                 my_party_id=my_party_id,
                                                                                 partner_party_id=partner_party_id,
                                                                                 my_role=my_role,
                                                                                 commit_id=commit_id,
                                                                                 tag=tag,
                                                                                 branch=branch)
    if commit_id:
        data_table_info = data_table_info if data_table_info else gen_data_table_info("feature_data", scene_key=scene_key, commit_id=commit_id)
        return get_data_table(data_table_info=data_table_info, create_if_missing=False)
    else:
        return None

if __name__ == '__main__':
    from arch.api import eggroll
    import uuid
    job_id = str(uuid.uuid1().hex)
    eggroll.init(job_id=job_id)

    commit_id = save_feature_header({"k1": 0, "k2": 1}, {"label1": 5})
    print(read_feature_header())

    save_feature_meta({"k1": 0, "k2": 1})
    print(read_feature_meta())
