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

from arch.api.proto.model_meta_pb2 import ModelMeta
from arch.api.proto.model_param_pb2 import ModelParam
from arch.api.proto.data_transform_pb2 import DataTransform
from arch.api.utils.core import get_fate_uuid, get_scene_key, get_commit_id, json_dumps, json_loads, \
    current_timestamp, string_bytes , bytes_string
from arch.api import eggroll
from arch.api import federation
from arch.api.io.table import gen_data_table_info, get_version_table, get_data_table, save_version_info

#scene_id, my_party_id, partner_party_id, my_role = 50001, 100001, 100001, "host"


def save_model(buffer_type, proto_buffer, commit_log="", branch="master"):
    version_table, data_table_info, scene_key, parent, commit_id = get_version_table("model_version", branch=branch, new_commit_id=True)
    data_table_info = gen_data_table_info("model_data", scene_key=scene_key, commit_id=commit_id)
    data_table = get_data_table(data_table_info=data_table_info, create_if_missing=True)
    # todo:  model slice?
    data_table.put(buffer_type, proto_buffer.SerializeToString(), use_pickle=False)

    # save model version info
    data_table_info["commitLog"] = commit_log
    data_table_info["parent"] = parent
    save_version_info(version_table=version_table, branch=branch, commit_id=commit_id, data_table_info=data_table_info)
    return commit_id


def read_model(buffer_type, proto_buffer, commit_id=None, tag=None, branch="master"):
    version_table, data_table_info, scene_key, parent, commit_id = get_version_table("model_version", commit_id=commit_id, tag=tag, branch=branch, new_commit_id=False)
    if commit_id:
        # Maybe param commit id or get commit id by current branch commit
        data_table_info = data_table_info if data_table_info else gen_data_table_info("model_data", scene_key=scene_key, commit_id=commit_id)
        data_table = get_data_table(data_table_info=data_table_info, create_if_missing=False)
        # todo:  model slice?
        buffer_bytes = data_table.get(buffer_type, use_pickle=False)
        proto_buffer.ParseFromString(buffer_bytes)
        return True
    else:
        return False



if __name__ == '__main__':
    import uuid
    job_id = str(uuid.uuid1().hex)
    eggroll.init(job_id=job_id)
    # guest
    with open("host_runtime_conf.json") as conf_fr:
        runtime_conf = json_loads(conf_fr.read())
    federation.init(job_id=job_id, runtime_conf=runtime_conf)
    print(federation.get_field("role"))

    model_meta_save = ModelMeta()
    model_meta_save.name = "HeteroLRGuest"
    commit_id = save_model("model_meta", model_meta_save, commit_log="xxx")
    print("save guest model success, commit id is %s" % commit_id)

    model_meta_read = ModelMeta()
    read_model("model_meta", model_meta_read)
    print(model_meta_read)

    model_param_save = ModelParam()
    model_param_save.weight["k1"] = 1
    model_param_save.weight["k2"] = 2
    commit_id = save_model("model_param", model_param_save, commit_log="xxx")
    print("save guest model success, commit id is %s" % commit_id)

    # read
    model_param_read = ModelParam()
    read_model("model_param", model_param_read)
    print(model_param_read)

    data_transform_save = DataTransform()
    data_transform_save.missing_replace_method = "test_method"
    save_model("data_transform", data_transform_save)

    data_transform_read = DataTransform()
    read_model("data_transform", data_transform_read)
    print(data_transform_read)
