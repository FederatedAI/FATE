#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
DEFAULT_ID_DELIMITER = ","


class StorageTableOrigin(object):
    TABLE_BIND = "table_bind"
    READER = "reader"
    UPLOAD = "upload"
    OUTPUT = "output"


class StorageEngine(object):
    STANDALONE = 'STANDALONE'
    EGGROLL = 'EGGROLL'
    HDFS = 'HDFS'
    MYSQL = 'MYSQL'
    SIMPLE = 'SIMPLE'
    PATH = 'PATH'
    HIVE = 'HIVE'
    LINKIS_HIVE = 'LINKIS_HIVE'
    LOCALFS = 'LOCALFS'


class StandaloneStoreType(object):
    ROLLPAIR_IN_MEMORY = 'IN_MEMORY'
    ROLLPAIR_LMDB = 'LMDB'
    DEFAULT = ROLLPAIR_LMDB


class EggRollStoreType(object):
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


class HDFSStoreType(object):
    RAM_DISK = 'RAM_DISK'
    SSD = 'SSD'
    DISK = 'DISK'
    ARCHIVE = 'ARCHIVE'
    DEFAULT = None


class PathStoreType(object):
    PICTURE = 'PICTURE'


class FileStoreType(object):
    CSV = 'CSV'


class MySQLStoreType(object):
    InnoDB = "InnoDB"
    MyISAM = "MyISAM"
    ISAM = "ISAM"
    HEAP = "HEAP"
    DEFAULT = None


class HiveStoreType(object):
    DEFAULT = "HDFS"


class LinkisHiveStoreType(object):
    DEFAULT = "HDFS"


class LocalFSStoreType(object):
    RAM_DISK = 'RAM_DISK'
    SSD = 'SSD'
    DISK = 'DISK'
    ARCHIVE = 'ARCHIVE'
    DEFAULT = None


class StorageTableMetaType(object):
    ENGINE = "engine"
    TYPE = "type"
    SCHEMA = "schema"
    PART_OF_DATA = "part_of_data"
    COUNT = "count"
    PARTITIONS = "partitions"
