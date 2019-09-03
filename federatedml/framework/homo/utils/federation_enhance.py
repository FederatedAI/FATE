import io
import pickle

from arch.api import federation
from arch.api.utils import log_utils

MAX_FRAG_SIZE = 1024 * 1024 * 100
LOGGER = log_utils.getLogger()


def _ger_frag_tag(tag_base):
    return lambda frag_id: "{0}_part_{1}".format(tag_base, frag_id)


def remote_fragment(obj, name, tag, role=None, idx=-1):
    dumper = FragmentPickleDumper()
    dumper.dump(obj)
    dumper.remote(name=name, tag=tag, role=role, idx=idx)


def get_fragment(name, tag, idx=-1):
    tag_fn = _ger_frag_tag(tag)
    num = federation.get(name, tag_fn(0), idx)
    if isinstance(num, list):
        # todo: add assertion
        frags = [[] for _ in range(len(num))]
        num = num[0]
        for frag_id in range(num):
            parts = federation.get(name, tag_fn(frag_id + 1), idx)
            for i, frag in enumerate(frags):
                frag.append(parts[i])
    else:
        frags = []
        for frag_id in range(num):
            part = federation.get(name, tag_fn(frag_id + 1), idx)
            frags.append(part)
    return FragmentPickleLoader(frags)


class FragmentPickleDumper(object):
    class Writer(object):
        def __init__(self, inner, max_frag_size):
            self._inner = inner
            self._max_frag_size = max_frag_size
            self._frag_space_remain = self._max_frag_size
            self._inner.append(bytearray())

        def write(self, write_bytes):
            end = len(write_bytes)
            start = 0
            while start < end:
                size = min(end - start, self._frag_space_remain)
                self._inner[-1].extend(write_bytes[start:start + size])
                start += size
                self._frag_space_remain -= size
                if self._frag_space_remain <= 0:
                    self._inner.append(bytearray())
                    self._frag_space_remain = self._max_frag_size

    def __init__(self, max_frag_size: int = MAX_FRAG_SIZE):
        self._inner = []
        self._writer = FragmentPickleDumper.Writer(self._inner, max_frag_size)

    def dump(self, obj):
        pickle.dump(obj, self._writer)

    def remote(self, name, tag, role=None, idx=-1):
        tag_fn = _ger_frag_tag(tag)
        num_frag = len(self._inner)
        federation.remote(num_frag, name, tag_fn(0), role, idx)
        for row_id, part in enumerate(self._inner):
            federation.remote(part, name, tag_fn(row_id + 1), role, idx)

    def merge(self):
        temp = bytearray()
        for frag in self._inner:
            temp.extend(frag)
        return bytes(temp)

    def num_frag(self):
        return len(self._inner)

    def __str__(self):
        return str(self._inner)

    def get_inner(self):
        return self._inner


# todo: implement read & readline directly
class FragmentPickleLoader(object):
    class Reader(object):
        def __init__(self, inner):
            self._inner = inner
            tmp = []
            for frag in self._inner:
                tmp.extend(frag)
            self._io = io.BytesIO(bytes(tmp))

        def readline(self):
            return self._io.readline()

        def read(self, size: int = 1):
            return self._io.read(size)

    def __init__(self, inner):
        self._inner = inner
        if isinstance(inner[0], list):
            self._readers = [
                FragmentPickleLoader.Reader(inn) for inn in self._inner
            ]
        else:
            self._readers = FragmentPickleLoader.Reader(self._inner)

    def load(self):
        if isinstance(self._readers, list):
            return [pickle.load(reader) for reader in self._readers]
        else:
            return pickle.load(self._readers)
