import logging

from ..abc.module import HeteroModule
from fate.interface import Context

logger = logging.getLogger(__name__)


class RawIntersectionGuest(HeteroModule):
    def __init__(self):
        ...

    def fit(self, ctx: Context, train_data, validate_data=None):
        # ctx.hosts.put("raw_index", train_data.index.tolist())
        ctx.hosts.put("raw_index", train_data.index.values)
        intersect_indexes = ctx.hosts.get("intersect_index")
        intersect_data = train_data
        for intersect_index in intersect_indexes:
            intersect_data = intersect_data.loc(intersect_index)

        return intersect_data


class RawIntersectionHost(HeteroModule):
    def __init__(self):
        ...

    def fit(self, ctx: Context, train_data, validate_data=None):
        guest_index = ctx.guest.get("raw_index")
        intersect_data = train_data.loc(guest_index)
        # ctx.guest.put("intersect_index", intersect_data.index.tolist())
        ctx.guest.put("intersect_index", intersect_data.index.values)

        return intersect_data
