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
import functools

from fate.arch.dataframe import DataFrame
from fate_utils.psi import Curve25519


GUEST_FIRST_SIGN = "guest_first_sign"
HOST_FIRST_SIGN = "host_first_sign"
GUEST_SECOND_SIGN = "guest_second_sign"
HOST_INDEXER = "host_indexer"


def _encrypt_bytes(values, curve: Curve25519 = None):
    id_list = [bytes(_id, "utf8") for _id in values[0]]
    return curve.encrypt_vec(id_list)


def _diffie_hellman(values, curve: Curve25519 = None):
    return curve.diffie_hellman_vec(values)


def _flat_block_with_possible_duplicate_keys(block_table, duplicate_allow=False):
    def _mapper(kvs):
        for partition_id, id_list in kvs:
            for _i, _id in enumerate(id_list):
                yield _id, [(partition_id, _i)]

    def _reducer(v1, v2):
        if not v1:
            return v2
        if not v2:
            return v1

        if not duplicate_allow:
            raise ValueError("duplicate match_id detect")

        return v1 + v2

    return block_table.mapReducePartitions(_mapper, _reducer)


def psi_ecdh(ctx, df: DataFrame, curve_type="curve25519", **kwargs):
    if curve_type != "curve25519":
        raise ValueError(f"Only support curve25519, curve_type={curve_type} is not implemented yet")

    if ctx.is_on_guest:
        return guest_run(ctx, df, curve_type, **kwargs)
    else:
        return host_run(ctx, df, curve_type, **kwargs)


def guest_run(ctx, df: DataFrame, curve_type="curve25519", **kwargs):
    curve = Curve25519()
    match_id = df.match_id.block_table

    encrypt_func = functools.partial(_encrypt_bytes, curve=curve)
    guest_first_sign_match_id = match_id.mapValues(encrypt_func)
    ctx.hosts.put(GUEST_FIRST_SIGN, guest_first_sign_match_id)

    host_first_sign_match_ids = ctx.hosts.get(HOST_FIRST_SIGN)
    host_second_sign_match_ids = []

    dh_func = functools.partial(_diffie_hellman, curve=curve)

    for i, host_first_sign_match_id in enumerate(host_first_sign_match_ids):
        host_second_sign_match_ids.append(
            _flat_block_with_possible_duplicate_keys(
                host_first_sign_match_ids[i].mapValues(dh_func),
                duplicate_allow=False
            )
        )

    guest_second_sign_match_ids = ctx.hosts.get(GUEST_SECOND_SIGN)

    intersect_id = None
    for guest_second_sign_id, host_second_sign_match_id in zip(guest_second_sign_match_ids, host_second_sign_match_ids):
        intersect_single = guest_second_sign_id.join(host_second_sign_match_id, lambda id_list_l, id_list_r: id_list_l)
        if not intersect_id:
            intersect_id = intersect_single
        else:
            intersect_id = intersect_id.join(intersect_single, lambda id_list_l, id_list_r: id_list_l)

    guest_df, new_indexer = df.iloc(intersect_id, return_new_indexer=True)
    """
    new_indexer: (intersect_id, [(sample_id, bid, offset) ...])
    """
    for host_id, host_second_sign_match_id in enumerate(host_second_sign_match_ids):
        host_indexer = host_second_sign_match_id.join(new_indexer, lambda v1, v2: (v1[0], v2))
        ctx.hosts[host_id].put(HOST_INDEXER, host_indexer)

    return guest_df


def host_run(ctx, df: DataFrame, curve_type, **kwargs):
    curve = Curve25519()
    match_id = df.match_id.block_table

    encrypt_func = functools.partial(_encrypt_bytes, curve=curve)
    host_first_sign_match_id = match_id.mapValues(encrypt_func)
    ctx.guest.put(HOST_FIRST_SIGN, host_first_sign_match_id)

    dh_func = functools.partial(_diffie_hellman, curve=curve)
    guest_first_sign_match_id = ctx.guest.get(GUEST_FIRST_SIGN)
    guest_second_sign_match_id = guest_first_sign_match_id.mapValues(dh_func)

    guest_second_sign_match_id = _flat_block_with_possible_duplicate_keys(guest_second_sign_match_id,
                                                                          duplicate_allow=True)
    ctx.guest.put(GUEST_SECOND_SIGN, guest_second_sign_match_id)

    host_indexer = ctx.guest.get(HOST_INDEXER)

    host_df = df.loc_with_sample_id_replacement(host_indexer)

    return host_df
