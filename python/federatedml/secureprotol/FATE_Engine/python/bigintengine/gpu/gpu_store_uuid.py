import uuid
from python.bigintengine.gpu.gpu_store_factory import gpu_store_factory

class Store_uuid:
    def __init__(self, store_uuid: uuid.UUID):
        if isinstance(store_uuid, uuid.UUID) is False:
            raise RuntimeError(f"Illegal store_uuid type : {type(store_uuid)}, params need type : {type(uuid.UUID)}")
        self.store_uuid = store_uuid

    def __del__(self):
        gpu_store_factory.release_store_status(self.store_uuid)
    
    @staticmethod
    def bi_alloc(vec_size: int):
        '''
            malloc bi memory from gpu_store_factory
        '''
        return Store_uuid(gpu_store_factory.bi_alloc(vec_size))

    @staticmethod
    def te_alloc(vec_size: int):
        '''
            malloc te memory from gpu_store_factory
        '''
        return Store_uuid(gpu_store_factory.te_alloc(vec_size))

    @staticmethod
    def fp_alloc(vec_size: int):
        '''
            malloc fp memory from gpu_store_factory
        '''
        return Store_uuid(gpu_store_factory.fp_alloc(vec_size))

    @staticmethod
    def pi_alloc(vec_size: int):
        '''
            malloc pi memory from gpu_store_factory
        '''
        return Store_uuid(gpu_store_factory.pi_alloc(vec_size))

    def get_store(self):
        '''
            get store from gpu_store_factory
        '''
        return gpu_store_factory.get_store(self.store_uuid)

PEN_store_uuid = Store_uuid
FPN_store_uuid = Store_uuid
TE_store_uuid = Store_uuid