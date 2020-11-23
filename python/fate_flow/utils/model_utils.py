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
import operator

from fate_flow.db.db_models import DB, MachineLearningModelInfo as MLModel

gen_key_string_separator = '#'


def gen_party_model_id(model_id, role, party_id):
    return gen_key_string_separator.join([role, str(party_id), model_id]) if model_id else None


def gen_model_id(all_party):
    return gen_key_string_separator.join([all_party_key(all_party), "model"])


def all_party_key(all_party):
    """
    Join all party as party key
    :param all_party:
        "role": {
            "guest": [9999],
            "host": [10000],
            "arbiter": [10000]
         }
    :return:
    """
    if not all_party:
        all_party_key = 'all'
    elif isinstance(all_party, dict):
        sorted_role_name = sorted(all_party.keys())
        all_party_key = gen_key_string_separator.join([
            ('%s-%s' % (
                role_name,
                '_'.join([str(p) for p in sorted(set(all_party[role_name]))]))
             )
            for role_name in sorted_role_name])
    else:
        all_party_key = None
    return all_party_key


@DB.connection_context()
def query_job(reverse=None, order_by=None, **kwargs):
    filters = []
    for f_n, f_v in kwargs.items():
        attr_name = 'f_%s' % f_n
        if hasattr(MLModel, attr_name):
            filters.append(operator.attrgetter('f_%s' % f_n)(MLModel) == f_v)
    if filters:
        models = MLModel.select().where(*filters)
        if reverse is not None:
            if not order_by or not hasattr(MLModel, f"f_{order_by}"):
                order_by = "create_time"
            if reverse is True:
                models = models.order_by(getattr(MLModel, f"f_{order_by}").desc())
            elif reverse is False:
                models = models.order_by(getattr(MLModel, f"f_{order_by}").asc())
        return [model for model in models]
    else:
        # not allow query all models
        return []

