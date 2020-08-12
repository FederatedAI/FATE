from enum import IntEnum, Enum


class WorkMode(IntEnum):
    STANDALONE = 0
    CLUSTER = 1

    def is_standalone(self):
        return self.value == self.STANDALONE

    def is_cluster(self):
        return self.value == self.CLUSTER


class Backend(IntEnum):
    EGGROLL = 0
    SPARK = 1
    STANDALONE = 2

    def is_spark(self):
        return self.value == self.SPARK

    def is_eggroll(self):
        return self.value == self.EGGROLL

    def is_standalone(self):
        return self.value == self.STANDALONE


class Party(object):
    """
    Uniquely identify
    """

    def __init__(self, role, party_id):
        self.role = role
        self.party_id = party_id

    def __hash__(self):
        return (self.role, self.party_id).__hash__()

    def __str__(self):
        return f"Party(role={self.role}, party_id={self.party_id})"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return (self.role, self.party_id) < (other.role, other.party_id)

    def __eq__(self, other):
        return self.party_id == other.party_id and self.role == other.role


class StandaloneStorageType(Enum):
    ROLLPAIR_IN_MEMORY = 'IN_MEMORY'
    ROLLPAIR_LMDB = 'LMDB'
    DEFAULT = ROLLPAIR_LMDB


class EggRollStorageType(Enum):
    ROLLPAIR_IN_MEMORY = 'IN_MEMORY'
    ROLLPAIR_LMDB = 'LMDB'
    ROLLPAIR_LEVELDB = 'LEVEL_DB'
    ROLLFRAME_FILE = 'ROLL_FRAME_FILE'
    ROLLPAIR_ROLLSITE = 'ROLL_SITE'
    ROLLPAIR_FILE = 'ROLL_PAIR_FILE'
    ROLLPAIR_MMAP = 'ROLL_PAIR_MMAP'
    ROLLPAIR_CACHE = 'ROLL_PAIR_CACHE'
    ROLLPAIR_QUEUE = 'ROLL_PAIR_QUEUE'
    DEFAULT = ROLLPAIR_LMDB


class HDFSStorageType(Enum):
    RAM_DISK = 'RAM_DISK'
    SSD = 'SSD'
    DISK = 'DISK'
    ARCHIVE = 'ARCHIVE'
    DEFAULT = None


class MySQLStorageType(Enum):
    InnoDB = "InnoDB"
    MyISAM = "MyISAM"
    ISAM = "ISAM"
    HEAP = "HEAP"
    DEFAULT = None


class StorageEngine(object):
    STANDALONE = 'STANDALONE'
    EGGROLL = 'EGGROLL'
    HDFS = 'HDFS'
    MYSQL = 'MYSQL'
    SIMPLE = 'SIMPLE'


class StorageTableMetaType(object):
    SCHEMA = "schema"
    PART_OF_DATA = "part_of_data"
    COUNT = "count"
    PARTITIONS = "partitions"
