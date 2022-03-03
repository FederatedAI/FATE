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


def generate_anonymous(fid, party_id=None, role=None, model=None):
    if model is None:
        if party_id is None or role is None:
            raise ValueError("party_id or role should be provided when generating"
                             "anonymous.")
    if party_id is None:
        party_id = model.component_properties.local_partyid
    if role is None:
        role = model.role

    party_id = str(party_id)
    fid = str(fid)
    return "_".join([role, party_id, fid])


def reconstruct_fid(encoded_name):
    try:
        col_index = int(encoded_name.split('_')[-1])
    except (IndexError, ValueError):
        raise RuntimeError(f"Decode name: {encoded_name} is not a valid value")
    return col_index
