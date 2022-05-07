# This file, together with gpu_store_uuid, new_gpu_store
# Form a way to eliminate the demand for re-malloc memory space 
# whe performing calculators. 

# Pros: 
#    1. Can accelerate the encrypted calculation by removing mallc time
#    2. A integreated memory management center
# Consts:
#    No free method, the memory space can only grow! 
#    May lead to severe memory leakage if not handled properly
#    If there is a burst of memory usage followed by a long plaintext compute
#    Then lots of memory may remain unused and unfreed
#    And datas are preserved unchanged and unfreed


import threading, uuid
from python.bigintengine.gpu.gpu_engine import bi_alloc, te_alloc, fp_alloc, pi_alloc, PLAIN_BYTE, MEM_HOST

class GPU_store_factory(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        self._bi_size_dict = {}
        self._te_size_dict = {}
        self._fp_size_dict = {}
        self._pi_size_dict = {}
        self._uuid_store_dict = {}
        self._uuid_status_dict = {}
        self._lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(GPU_store_factory, "_instance"):
            with GPU_store_factory._instance_lock:
                if not hasattr(GPU_store_factory, "_instance"):
                    GPU_store_factory._instance = object.__new__(cls)
        return GPU_store_factory._instance

    # bi_size_dict : 0
    # te_size_dict : 1
    # fp_size_dict : 2
    # pi_size_dict : 3
    def _solve_alloc(self, dict_id: int, vec_size: int):
        if dict_id == 0:
            cur_dict = self._bi_size_dict
            cur_func = bi_alloc
        elif dict_id == 1:
            cur_dict = self._te_size_dict
            cur_func = te_alloc
        elif dict_id == 2:
            cur_dict = self._fp_size_dict
            cur_func = fp_alloc
        elif dict_id == 3:
            cur_dict = self._pi_size_dict
            cur_func = pi_alloc
        else:
            raise RuntimeError("finding dict_id not existing!")

        if cur_dict.get(vec_size) is None:
            cur_dict[vec_size] = []
        
        # find whether there is a free uuid
        # aka whether there is a free memory space of desired size
        for cur_uuid in cur_dict[vec_size]:
            if self._uuid_status_dict[cur_uuid] is False:
                self._uuid_status_dict[cur_uuid] = True
                return cur_uuid
        # no suitable uuid, then create a new one
        cur_uuid = uuid.uuid1()
        cur_dict[vec_size].append(cur_uuid)
        if dict_id == 0:
            self._uuid_store_dict[cur_uuid] = cur_func(None, vec_size, PLAIN_BYTE, MEM_HOST)
            self._uuid_status_dict[cur_uuid] = True
        else:
            self._uuid_store_dict[cur_uuid] = cur_func(None, vec_size, MEM_HOST)
            self._uuid_status_dict[cur_uuid] = True
        return cur_uuid

    def get_store(self, cur_uuid: str):
        if self._uuid_store_dict.get(cur_uuid) is None:
            raise RuntimeError("Getting store not existing!")
        return self._uuid_store_dict[cur_uuid]

    def bi_alloc(self, vec_size: int):
        self._lock.acquire()
        try:
            return self._solve_alloc(0, vec_size)
        finally:
            self._lock.release()

    def te_alloc(self, vec_size: int):
        self._lock.acquire()
        try:
            return self._solve_alloc(1, vec_size)
        finally:
            self._lock.release()
    
    def fp_alloc(self, vec_size: int):
        self._lock.acquire()
        try:
            return self._solve_alloc(2, vec_size)
        finally:
            self._lock.release()

    def pi_alloc(self, vec_size: int):
        self._lock.acquire()
        try:
            return self._solve_alloc(3, vec_size)
        finally:
            self._lock.release()

    def release_store_status(self, cur_uuid: str):
        if self._uuid_status_dict.get(cur_uuid) is None:
            raise RuntimeError("Releasing store not existing!")
        self._uuid_status_dict[cur_uuid] = False
    
    def get_total_size(self):
        return len(self._uuid_status_dict)

    def get_free_size(self):
        count = 0
        for i in self._uuid_status_dict:
            count = count + 1 if self._uuid_status_dict[i] is False else count
        return count

    def get_uuid_dict(self):
        return self._uuid_store_dict

gpu_store_factory = GPU_store_factory()
