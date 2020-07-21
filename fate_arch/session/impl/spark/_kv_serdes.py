import pickle

from pyspark import RDD

from fate_arch.session.impl.spark._util import _generate_hdfs_path, _get_file_system, _get_path

_DELIMITER = '\t'


def _deserialize(m):
    fields = m.strip().partition(_DELIMITER)
    return fields[0], pickle.loads(bytes.fromhex(fields[2]))


def _serialize(k, v):
    return f"{k}{_DELIMITER}{pickle.dumps(v).hex()}"


def _partition_deserialize(it):
    for m in it:
        yield _deserialize(m)


def _load_from_hdfs(sc, paths, partitions):
    fs = _get_file_system(sc)
    path = _get_path(sc, paths)

    if not fs.exists(path):
        raise FileNotFoundError(f"{paths} not found")
    return sc.textFile(path, partitions).mapPartitions(_partition_deserialize).repartition(partitions)


def _save_as_hdfs(rdd: RDD, namespace, name):
    sc = rdd.context
    hdfs_path = _generate_hdfs_path(namespace=namespace, name=name)
    fs = _get_file_system(sc)
    path = _get_path(sc, hdfs_path)
    if fs.exists(path):
        raise FileExistsError(f"{namespace}/{name} exist")

    rdd.map(lambda x: _serialize(x[0], x[1])).saveAsTextFile(path)
