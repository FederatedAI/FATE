from federatedml.util import consts
from fate_arch.session import computing_session as session
from federatedml.util import LOGGER

class Guest():
    def __init__(self):
        self.mini_batch_obj = None #一个MiniBatch对象，负责实际的数据的拆分
        self.batch_nums = None #多少批数据

    def register_batch_generator(self,transfer_variables):
        """
        batch_data_index_transfer：用来传输数据对应的ID，便于双方进行同步数据
        batch_info：主要用来传俩信息，其一是多少批数据；其二是每批多少数据
        """
        self.batch_data_index_transfer = transfer_variables.batch_data_index
        self.batch_info = transfer_variables.batch_info

    def sync_batch_index(self,batch_index,suffix=tuple()):
        self.batch_data_index_transfer.remote(obj=batch_index,role=consts.HOST,suffix=suffix)

    def sync_batch_info(self,batch_info,suffix=tuple()):
        #batch_info本身就是一个字典迭代类型
        self.batch_info.remote(obj=batch_info,role=consts.HOST,suffix=suffix)

    def initialize_batch_generator(self,data_instances,batch_size,suffix=tuple()):
        """
        这个函数主要做俩事情：
        其一：拆分数据
        其二：将拆分的若干批数据的ID传给host，以便于Host方进行同步
        """
        self.mini_batch_obj = MiniBatch(data_insts=data_instances,batch_size=batch_size)
        self.batch_nums = self.mini_batch_obj.batch_nums

        LOGGER.info("一共多少批数据：{},每批数据量为:{}".format(self.batch_nums,batch_size))
        batch_info = {"batch_size": batch_size, "batch_nums": self.batch_nums}
        self.sync_batch_info(batch_info,suffix=suffix)

        index_generator = self.mini_batch_obj.mini_batch_data_generator(index="index")
        batch_index = 0
        for batch_data_index in index_generator:
            #这里纳闷了好久，元组不能增加元素，同时元组添加新元素，逗号不能省略
            batch_suffix = suffix + (batch_index,)
            #将当前批次的数据的索引发送给对方
            self.sync_batch_index(batch_data_index,batch_suffix)
            batch_index += 1

    def generator_batch_data(self):
        #实际生成数据，使用一个不常用的关键字：yield。节省内存
        data_generator = self.mini_batch_obj.mini_batch_data_generator(index="data")
        for batch_data in data_generator:
            yield batch_data


class Host():
    def __init__(self):
        self.batch_data_insts = []
        self.batch_nums = None

    def register_batch_generator(self,transfer_variables):
        """
        batch_data_index_transfer：用来接收数据对应的ID，便于双方进行同步数据
        batch_info：主要用来接受信息，其一是多少批数据；其二是每批多少数据
        """
        self.batch_data_index_transfer = transfer_variables.batch_data_index
        self.batch_info = transfer_variables.batch_info

    def sync_batch_index(self,suffix=tuple()):
        #这里的get直接得到的就是字典，不用列表封装了。这里的官方API文档是错的
        batch_index = self.batch_data_index_transfer.get(suffix=suffix)
        return batch_index

    def sync_batch_info(self,suffix=tuple()):
        LOGGER.debug("在这个信息中，suffix是：{}".format(suffix))
        batch_info = self.batch_info.get(suffix=suffix)
        #对象是迭代类型的时候，对象类型不需要转换为list
        batch_size = batch_info[0].get("batch_size")
        if batch_size < consts.MIN_BATCH_SIZE and batch_size != -1:
            raise ValueError("当前的batch_size太小了，值为:{}".format(batch_size))
        return batch_info

    def initialize_batch_generator(self,data_inst,suffix=tuple()):
        batch_info = self.sync_batch_info(suffix)
        LOGGER.info("从guest方接收到的batch_info信息是：{}".format(batch_info))
        self.batch_nums = batch_info[0].get("batch_nums")

        for batch_index in range(self.batch_nums):
            batch_suffix = suffix + (batch_index,)
            batch_data_index = self.sync_batch_index(batch_suffix)
            batch_data_inst = batch_data_index[0].join(data_inst,lambda x,y : y)
            self.batch_data_insts.append(batch_data_inst)

    def generator_batch_data(self):
        batch_index = 0

        for batch_data_inst in self.batch_data_insts:
            LOGGER.info("当前的batch_num是:{}，batch_data_inst size是:{}".format(batch_index,batch_data_inst.count()) )
            yield batch_data_inst
            batch_index += 1


class MiniBatch():
    """
    这个部分主要用来生成小批量数据
    """
    def __init__(self,data_insts,batch_size):
        self.all_batch_data = None #用来存批量数据，数据结构是列表，元素是Dtable
        self.all_index_data = None #用来存批量数据对应的索引，数据结构是列表，元素是Dtable
        self.data_insts = data_insts
        self.batch_nums = 0
        self.batch_size = batch_size

        self.__mini_batch_data_seperator(data_insts,self.batch_size)

    def __mini_batch_data_seperator(self,data_insts,batch_size):
        """
        这个函数实际就是对以下两个属性进行赋值操作：
        self.all_batch_data\self.all_index_data
        """
        data_ids_iter,data_size = self.collect_index(data_insts)
        if self.batch_size > data_size:
            self.batch_size = data_size
            batch_size = data_size
        #之所以这么写是为了保证全部数据都来一波训练
        batch_nums = (data_size + batch_size -1) // batch_size

        batch_data_ids = [] #用来存最终的拆分好的数据的ID，数据结构是一个列表，元素就是current_batch_ids，
        current_data_nums = 0 #用来统计当前遍历到哪一条数据了
        current_batch = 0 #当前是属于哪一批数据
        current_batch_ids = [] #当前批次的数据，元素是元素。元组的第一项是ID，第二项是None

        #这里的data_ids_iter是一个generator。其中的values实际用不到，所以设置为None
        for ids,values in data_ids_iter:
            #数据的所有ID都会以元组为单位存在列表中
            current_batch_ids.append((ids,None))
            #数据总量
            current_data_nums += 1
            #当正好够一波数据后，存到batch_data_ids当中
            if current_data_nums % batch_size == 0:
                current_batch += 1
                if current_batch < batch_nums:
                    batch_data_ids.append(current_batch_ids)
                    current_batch_ids = []
            #后续会有剩余的一批数据不够一波，但是还是要把它当成一波存起来
            if current_data_nums == data_size and len(current_batch_ids) != 0:
                batch_data_ids.append(current_batch_ids)

        self.batch_nums = len(batch_data_ids)
        all_batch_data = []
        all_index_data = []
        for index_data in batch_data_ids:
            index_table = session.parallelize(index_data,partition=data_insts.partitions,include_key=True)
            batch_data = index_table.join(data_insts,lambda x,y:y)

            #这里的index_data、batch_data都是Table格式的数据
            all_batch_data.append(batch_data)
            all_index_data.append(index_table)
        self.all_batch_data = all_batch_data
        self.all_index_data = all_index_data

    def mini_batch_data_generator(self,index = "data"):
        """
        生成小批量数据。可以选择生成索引或者索引对应的数据
        """
        LOGGER.debug("一共有多少批量数据：{}".format(len(self.all_batch_data)))
        if index == "data":
            for batch_data in self.all_batch_data:
                yield batch_data
        else:
            for index_data in self.all_index_data:
                yield index_data

    def collect_index(self,data_insts):
        #这里只选取数据的ID
        data_ids = data_insts.mapValues(lambda x : None)
        data_size = data_insts.count()
        #下面这个玩意是一个generator
        data_ids_iter = data_ids.collect()
        data_ids_iter = sorted(data_ids_iter,key=lambda x : x[0])

        return data_ids_iter,data_size

# class TestTransfer():
#     def __init__(self):
#         self.ir_a = None
#         self.ir_b = None
#
#     def initialize(self,transfer_variable):
#         self.ir_a = transfer_variable.ir_a
#         self.ir_b = transfer_variable.ir_b
#
#     def send(self,obj,role,suffix,):
#         if role == "guest":
#             self.ir_b.remote(obj=obj,role=role,suffix=suffix)
#         else:
#             self.ir_a.remote(obj=obj,role=role,suffix=suffix)
#
#     def receive(self,suffix):
#         pass
