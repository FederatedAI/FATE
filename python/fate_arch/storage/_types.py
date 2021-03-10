from fate_arch.computing import ComputingEngine
from fate_arch.common.address import StandaloneAddress, EggRollAddress, HDFSAddress, MysqlAddress, FileAddress, \
    PathAddress


class StorageEngine(object):
    STANDALONE = 'STANDALONE'
    EGGROLL = 'EGGROLL'
    HDFS = 'HDFS'
    MYSQL = 'MYSQL'
    SIMPLE = 'SIMPLE'
    FILE = 'FILE'
    PATH = 'PATH'
    LOCAL = 'LOCAL'


class StandaloneStorageType(object):
    ROLLPAIR_IN_MEMORY = 'IN_MEMORY'
    ROLLPAIR_LMDB = 'LMDB'
    DEFAULT = ROLLPAIR_LMDB


class EggRollStorageType(object):
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


class LocalStorageType(object):
    FILE = 'FILE'


class HDFSStorageType(object):
    RAM_DISK = 'RAM_DISK'
    SSD = 'SSD'
    DISK = 'DISK'
    ARCHIVE = 'ARCHIVE'
    DEFAULT = None


class PathStorageType(object):
    PICTURE = 'PICTURE'


class FileStorageType(object):
    CSV = 'CSV'


class MySQLStorageType(object):
    InnoDB = "InnoDB"
    MyISAM = "MyISAM"
    ISAM = "ISAM"
    HEAP = "HEAP"
    DEFAULT = None


class StorageTableMetaType(object):
    ENGINE = "engine"
    TYPE = "type"
    SCHEMA = "schema"
    PART_OF_DATA = "part_of_data"
    COUNT = "count"
    PARTITIONS = "partitions"


class Relationship(object):
    CompToStore = {
        ComputingEngine.STANDALONE: [StorageEngine.STANDALONE],
        ComputingEngine.EGGROLL: [StorageEngine.EGGROLL],
        ComputingEngine.SPARK: [StorageEngine.HDFS, StorageEngine.LOCAL]
    }
    EngineToAddress = {
        StorageEngine.STANDALONE: StandaloneAddress,
        StorageEngine.EGGROLL: EggRollAddress,
        StorageEngine.HDFS: HDFSAddress,
        StorageEngine.MYSQL: MysqlAddress,
        StorageEngine.FILE: FileAddress,
        StorageEngine.PATH: PathAddress
        StorageEngine.LOCAL: StandaloneAddress
    }


DEFAULT_ID_DELIMITER = ","
