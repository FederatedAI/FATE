import functools
import typing

from fate_arch.common import Party
from fate_arch.federation._gc import IterationGC
from fate_arch.common._parties import Role, PartiesInfo
from fate_arch.common.log import getLogger

LOGGER = getLogger()


class _TagGlobalContext:
    _contexts = []

    @classmethod
    def set(cls, context):
        cls._contexts.append(context)

    @classmethod
    def reset(cls, tag):
        popped = cls._contexts.pop()
        if tag != popped:
            raise RuntimeError(
                f"try to exit context tag: {tag}, find {popped}. Make sure use context in `First Enter, Last Exit` order"
            )

    @classmethod
    def get(cls):
        if cls._contexts:
            return cls._contexts[-1]
        return None


class Tag:
    def __init__(self, tag):
        self.tag = tag

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    def open(self):
        _TagGlobalContext.set(self.tag)
        return self

    def close(self):
        return _TagGlobalContext.reset(self.tag)

    def __call__(self, func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        def _wrap(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return _wrap


class _TagSpace:
    def __init__(self) -> None:
        self._tags = {}
        self._gc = {}

    def get(self, name: str, party: Party):
        if name not in self._tags:
            self._tags[name] = {}
            self._gc[name] = {}

        if party not in self._tags[name]:
            self._tags[name][party] = 0
            self._gc[name][party] = IterationGC()

        return self._tags[name][party], self._gc[name][party]

    def inc(self, name: str, party: typing.Union[Party, typing.List[Party]]):
        if not isinstance(party, list):
            party = [party]
        for p in party:
            cur, _ = self.get(name, p)
            self._tags[name][p] = cur + 1

    def get_list(
            self, name: str, parties: typing.List[Party]
    ) -> typing.Tuple[typing.List[str], typing.List[IterationGC]]:
        tag_list = []
        gc_list = []
        for p in parties:
            tag, gc = self.get(name, p)
            tag_list.append(str(tag))
            gc_list.append(gc)
        return tag_list, gc_list


_REMOTE_TAG_SPACE = _TagSpace()
_GET_TAG_SPACE = _TagSpace()

_TYPE_PARTIES = typing.Union[
    typing.Union[Role, Party], typing.List[typing.Union[Role, Party]]
]


def remote(
        parties: _TYPE_PARTIES,
        name,
        v
):
    from fate_arch.session import get_session

    tag = _TagGlobalContext.get()
    if tag is not None:
        name = f"{tag}.{name}"

    parties = PartiesInfo.get_parties(parties)
    session = get_session()
    tags, gc_list = _REMOTE_TAG_SPACE.get_list(name=name, parties=parties)
    for p, t, gc in zip(parties, tags, gc_list):
        LOGGER.debug(f"remote name={name}, tag={t}")
        session.federation.remote(v=v, name=name, tag=t, parties=[p], gc=gc)
        gc.gc()
    _REMOTE_TAG_SPACE.inc(name=name, party=parties)


def get(parties: _TYPE_PARTIES, name):
    from fate_arch.session import get_session

    tag = _TagGlobalContext.get()
    if tag is not None:
        name = f"{tag}.{name}"

    return_single = isinstance(parties, Party)
    parties = PartiesInfo.get_parties(parties)
    session = get_session()
    LOGGER.debug(f"_GET_TAG_SPACE={_GET_TAG_SPACE._tags}")
    tags, gc_list = _GET_TAG_SPACE.get_list(name=name, parties=parties)
    objs = []
    for p, t, gc in zip(parties, tags, gc_list):
        LOGGER.debug(f"get name={name}, tag={t}")
        objs.extend(session.federation.get(name=name, tag=t, parties=[p], gc=gc))

    _GET_TAG_SPACE.inc(name=name, party=parties)

    if return_single:
        return objs[0]

    return objs