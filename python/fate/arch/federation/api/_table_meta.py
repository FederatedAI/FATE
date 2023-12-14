class TableMeta:
    def __init__(self, num_partitions: int, key_serdes_type: int, value_serdes_type: int, partitioner_type: int):
        self.num_partitions = num_partitions
        self.key_serdes_type = key_serdes_type
        self.value_serdes_type = value_serdes_type
        self.partitioner_type = partitioner_type
