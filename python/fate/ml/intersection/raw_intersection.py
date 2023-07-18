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
import logging

from fate.arch import Context

from ..abc.module import HeteroModule

logger = logging.getLogger(__name__)


class RawIntersectionGuest(HeteroModule):
    def __init__(self):
        ...

    def fit(self, ctx: Context, train_data, validate_data=None):
        # ctx.hosts.put("raw_index", train_data.index.tolist())
        ctx.hosts.put("raw_index", train_data.get_indexer(target="sample_id"))
        intersect_indexes = ctx.hosts.get("intersect_index")
        intersect_data = train_data
        for intersect_index in intersect_indexes:
            intersect_data = intersect_data.loc(intersect_index, preserve_order=True)

        intersect_count = intersect_data.count()
        ctx.hosts.put("intersect_count", intersect_count)

        logger.info(f"intersect count={intersect_count}")
        return intersect_data


class RawIntersectionHost(HeteroModule):
    def __init__(self):
        ...

    def fit(self, ctx: Context, train_data, validate_data=None):
        guest_index = ctx.guest.get("raw_index")
        intersect_data = train_data.loc(guest_index)
        # ctx.guest.put("intersect_index", intersect_data.index.tolist())
        ctx.guest.put("intersect_index", intersect_data.get_indexer(target="sample_id"))

        intersect_count = ctx.guest.get("intersect_count")
        logger.info(f"intersect count={intersect_count}")
        return intersect_data
