#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

import fate.arch.protocol.mpc.communicator as comm


class TupleProvider:
    TRACEABLE_FUNCTIONS = [
        "generate_additive_triple",
        "square",
        "generate_binary_triple",
        "wrap_rng",
        "B2A_rng",
    ]

    _DEFAULT_CACHE_PATH = os.path.normpath(os.path.join(__file__, "../tuple_cache/"))

    def __init__(self):
        self.tracing = False
        self.request_cache = []
        self.tuple_cache = {}

    @property
    def rank(self):
        return comm.get().get_rank()

    def _get_request_path(self, prefix=None):
        if prefix is None:
            prefix = self._DEFAULT_CACHE_PATH
        return prefix + f"/request_cache-{self.rank}"

    def _get_tuple_path(self, prefix=None):
        if prefix is None:
            prefix = self._DEFAULT_CACHE_PATH
        return prefix + f"/tuple_cache-{self.rank}"

    def trace(self, tracing=True):
        """Sets tracing attribute.

        When tracing is True, provider caches all tuple requests.
        When tracing is False, provider attempts to load tuples from cache.
        """
        self.tracing = tracing

    def trace_once(self):
        """Sets tracing attribute True only if the request cache is empty.
        If `trace_once()` is called again, it sets tracing attribute to False
        """
        untraced = self.request_cache.empty()
        self.trace(tracing=untraced)

    def _save_requests(self, filepath=None):
        # TODO: Deal with any overwrite issues
        if len(self.request_cache) == 0:
            crypten.log("Request cache not saved - cache is empty")
            return
        filepath = self._get_request_path(prefix=filepath)
        torch.save(self.request_cache, filepath)
        self.request_cache = []

    def _load_requests(self, filepath=None):
        filepath = self._get_request_path(prefix=filepath)
        if os.path.exists(filepath):
            self.request_cache = torch.load(filepath)
            os.remove(filepath)
        else:
            crypten.log(f"Cache requests not loaded - File `{filepath}` not found")

    def _save_tuples(self, filepath=None):
        # TODO: Deal with any overwrite issues
        if len(self.tuple_cache) == 0:
            crypten.log("Tuple cache not saved - cache is empty")
            return
        filepath = self._get_tuple_path(prefix=filepath)
        torch.save(self.tuple_cache, filepath)
        self.tuple_cache = {}

    def _load_tuples(self, filepath=None):
        filepath = self._get_tuple_path(prefix=filepath)
        if os.path.exists(filepath):
            self.tuple_cache = torch.load(filepath)
            os.remove(filepath)
        else:
            crypten.log(f"Tuple cache not loaded - File `{filepath}` not found")

    def save_cache(self, filepath=None):
        """Saves request and tuple cache to a file.

        args:
            filepath - base filepath for cache folder (default: "provider/tuple_cache/")
        """
        self._save_requests(filepath=filepath)
        self._save_tuples(filepath=filepath)

    def load_cache(self, filepath=None):
        """Loads request and tuple cache from a file.

        args:
            filepath - base filepath for cache folder (default: "provider/tuple_cache/")
        """
        self._load_requests(filepath=filepath)
        self._load_tuples(filepath=filepath)

    def __getattribute__(self, func_name):
        """Deals with caching logic"""
        if func_name not in TupleProvider.TRACEABLE_FUNCTIONS:
            return object.__getattribute__(self, func_name)

        # Trace requests while tracing
        if self.tracing:

            def func_with_trace(*args, **kwargs):
                request = (func_name, args, kwargs)
                self.request_cache.append(request)
                return object.__getattribute__(self, func_name)(*args, **kwargs)

            return func_with_trace

        # If the cache is empty, call function directly
        if len(self.tuple_cache) == 0:
            return object.__getattribute__(self, func_name)

        # Return results from cache if available
        def func_from_cache(*args, **kwargs):
            hashable_kwargs = frozenset(kwargs.items())
            request = (func_name, args, hashable_kwargs)
            # Read from cache
            if request in self.tuple_cache.keys():
                return self.tuple_cache[request].pop()
            # Cache miss
            return object.__getattribute__(self, func_name)(*args, **kwargs)

        return func_from_cache

    def fill_cache(self):
        """Fills tuple_cache with tuples requested in the request_cache"""
        # TODO: parallelize / async this
        for request in self.request_cache:
            func_name, args, kwargs = request
            result = object.__getattribute__(self, func_name)(*args, **kwargs)

            hashable_kwargs = frozenset(kwargs.items())
            hashable_request = (func_name, args, hashable_kwargs)
            if hashable_request in self.tuple_cache.keys():
                self.tuple_cache[hashable_request].append(result)
            else:
                self.tuple_cache[hashable_request] = [result]

    def generate_additive_triple(self, size0, size1, op, device=None, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        raise NotImplementedError("TupleProvider generate_additive_triple not implemented.")

    def square(self, size, device=None):
        """Generate square double of given size"""
        raise NotImplementedError("TupleProvider square not implemented.")

    def generate_binary_triple(self, size0, size1, device=None):
        """Generate xor triples of given size"""
        raise NotImplementedError("TupleProvider generate_binary_triple not implemented.")

    def wrap_rng(self, size, device=None):
        """Generate random shared tensor of given size and sharing of its wraps"""
        raise NotImplementedError("TupleProvider wrap_rng not implemented.")

    def B2A_rng(self, size, device=None):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        raise NotImplementedError("TupleProvider B2A_rng not implemented.")
