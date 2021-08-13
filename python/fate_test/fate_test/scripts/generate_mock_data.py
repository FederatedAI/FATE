import hashlib
import json
import os
import random
import threading
import sys
import time
import copy
import uuid
import functools
import pandas as pd
import numpy as np

from fate_arch import storage
from fate_arch.common import WorkMode, Backend
from fate_flow.utils import data_utils
from fate_arch.session import Session
from fate_arch.storage import StorageEngine
from fate_test._config import Config

sys.setrecursionlimit(1000000)


class data_progress:
    def __init__(self, down_load, time_start):
        self.time_start = time_start
        self.down_load = down_load
        self.time_percent = 0
        self.switch = True

    def set_switch(self, switch):
        self.switch = switch

    def get_switch(self):
        return self.switch

    def set_time_percent(self, time_percent):
        self.time_percent = time_percent

    def get_time_percent(self):
        return self.time_percent

    def progress(self, percent):
        if percent > 100:
            percent = 100
        end = time.time()
        if percent != 100:
            print(f"\r{self.down_load}  %.f%s  [%s]  running" % (percent, '%', self.timer(end - self.time_start)),
                  flush=True, end='')
        else:
            print(f"\r{self.down_load}  %.f%s  [%s]  success" % (percent, '%', self.timer(end - self.time_start)),
                  flush=True, end='')

    @staticmethod
    def timer(times):
        hours, rem = divmod(times, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


def remove_file(path):
    os.remove(path)


def id_encryption(encryption_type, start_num, end_num):
    if encryption_type == 'md5':
        return [hashlib.md5(bytes(str(value), encoding='utf-8')).hexdigest() for value in range(start_num, end_num)]
    elif encryption_type == 'sha256':
        return [hashlib.sha256(bytes(str(value), encoding='utf-8')).hexdigest() for value in range(start_num, end_num)]
    else:
        return [str(value) for value in range(start_num, end_num)]


def get_big_data(guest_data_size, host_data_size, guest_feature_num, host_feature_num, include_path, host_data_type,
                 conf: Config, encryption_type, match_rate, sparsity, force, split_host, output_path, parallelize):
    global big_data_dir

    def list_tag_value(feature_nums, head):
        data = ''
        for f in range(feature_nums):
            data += head[f] + ':' + str(round(np.random.randn(), 2)) + ";"
        return data[:-1]

    def list_tag(feature_nums, data_list):
        data = ''
        for f in range(feature_nums):
            data += random.choice(data_list) + ";"
        return data[:-1]

    def _generate_tag_value_data(data_path, start_num, end_num, feature_nums):
        data_num = end_num - start_num
        section_data_size = round(data_num / 100)
        iteration = round(data_num / section_data_size)
        head = ['x' + str(i) for i in range(feature_nums)]
        for batch in range(iteration + 1):
            progress.set_time_percent(batch)
            output_data = pd.DataFrame(columns=["id"])
            if section_data_size * (batch + 1) <= data_num:
                output_data["id"] = id_encryption(encryption_type, section_data_size * batch + start_num,
                                                  section_data_size * (batch + 1) + start_num)
                slicing_data_size = section_data_size
            elif section_data_size * batch <= data_num:
                output_data['id'] = id_encryption(encryption_type, section_data_size * batch + start_num, end_num)
                slicing_data_size = data_num - section_data_size * batch
            else:
                break
            feature = [list_tag_value(feature_nums, head) for i in range(slicing_data_size)]
            output_data['feature'] = feature
            output_data.to_csv(data_path, mode='a+', index=False, header=False)

    def _generate_dens_data(data_path, start_num, end_num, feature_nums, label_flag):
        if label_flag:
            head_1 = ['id', 'y']
        else:
            head_1 = ['id']
        data_num = end_num - start_num
        head_2 = ['x' + str(i) for i in range(feature_nums)]
        df_data_1 = pd.DataFrame(columns=head_1)
        head_data = pd.DataFrame(columns=head_1 + head_2)
        head_data.to_csv(data_path, mode='a+', index=False)
        section_data_size = round(data_num / 100)
        iteration = round(data_num / section_data_size)
        for batch in range(iteration + 1):
            progress.set_time_percent(batch)
            if section_data_size * (batch + 1) <= data_num:
                df_data_1["id"] = id_encryption(encryption_type, section_data_size * batch + start_num,
                                                section_data_size * (batch + 1) + start_num)
                slicing_data_size = section_data_size
            elif section_data_size * batch <= data_num:
                df_data_1["id"] = id_encryption(encryption_type, section_data_size * batch + start_num, end_num)
                slicing_data_size = data_num - section_data_size * batch
            else:
                break
            if label_flag:
                df_data_1["y"] = [round(np.random.random()) for x in range(slicing_data_size)]
            feature = np.random.randint(-100, 100, size=[slicing_data_size, feature_nums]) / 100
            df_data_2 = pd.DataFrame(feature, columns=head_2)
            output_data = pd.concat([df_data_1, df_data_2], axis=1)
            output_data.to_csv(data_path, mode='a+', index=False, header=False)

    def _generate_tag_data(data_path, start_num, end_num, feature_nums, sparsity):
        data_num = end_num - start_num
        section_data_size = round(data_num / 100)
        iteration = round(data_num / section_data_size)
        valid_set = [x for x in range(2019120799, 2019120799 + round(feature_nums / sparsity))]
        data = list(map(str, valid_set))
        for batch in range(iteration + 1):
            progress.set_time_percent(batch)
            output_data = pd.DataFrame(columns=["id"])
            if section_data_size * (batch + 1) <= data_num:
                output_data["id"] = id_encryption(encryption_type, section_data_size * batch + start_num,
                                                  section_data_size * (batch + 1) + start_num)
                slicing_data_size = section_data_size
            elif section_data_size * batch <= data_num:
                output_data["id"] = id_encryption(encryption_type, section_data_size * batch + start_num, end_num)
                slicing_data_size = data_num - section_data_size * batch
            else:
                break
            feature = [list_tag(feature_nums, data_list=data) for i in range(slicing_data_size)]
            output_data['feature'] = feature
            output_data.to_csv(data_path, mode='a+', index=False, header=False)

    def _generate_parallelize_data(start_num, end_num, feature_nums, table_name, namespace, path_prefix="data_test"):
        session = Session.create(work_mode=work_mode, backend=backend)
        data_num = end_num - start_num
        session_id = str(uuid.uuid1())
        session.init_computing(session_id)
        step = 10000 if data_num > 10000 else int(data_num / 10)
        table_list = [(f"{i * step}", f"{feature_nums}") for i in range(int(data_num / step) + start_num)]
        partition = 4
        table = session.computing.parallelize(table_list, partition=partition, include_key=True)

        def expand_id_range(k, v):
            if label_flag:
                return [(id_encryption(encryption_type, ids, ids + 1),
                         ",".join([str(round(np.random.random()))] + [str(i) for i in
                                                                      np.random.randint(-100, 100, size=int(v)) / 100]))
                        for ids in range(int(k), min(step + int(k), end_num))]
            else:
                if data_type == 'tag':
                    valid_set = [x for x in range(2019120799, 2019120799 + round(feature_nums / sparsity))]
                    data = list(map(str, valid_set))
                    return [(id_encryption(encryption_type, ids, ids + 1),
                             ";".join([random.choice(data) for i in range(int(v))]))
                            for ids in range(int(k), min(step + int(k), data_num))]
                elif data_type == 'tag_value':
                    return [(id_encryption(encryption_type, ids, ids + 1), ",".join(
                        [f"x{i}" + ':' + str(round(np.random.randn(), 2)) + ";" for i in
                         np.random.uniform(size=int(v))]))
                            for ids in range(int(k), min(step + int(k), data_num))]
                elif data_type == 'dense':
                    return [(id_encryption(encryption_type, ids, ids + 1),
                             ",".join([str(i) for i in np.random.randint(-100, 100, size=int(v)) / 100]))
                            for ids in range(int(k), min(step + int(k), data_num))]

        table = table.flatMap(functools.partial(expand_id_range))
        if label_flag:
            table.schema = {"sid": "id", "header": ",".join(["y"] + [f"x{i}" for i in range(feature_nums)])}
        else:
            table.schema = {"sid": "id", "header": ",".join([f"x{i}" for i in range(feature_nums)])}

        if work_mode == WorkMode.STANDALONE:
            address_dict = {"name": table_name, "namespace": namespace}
            storage_engine = StorageEngine.STANDALONE
        elif work_mode == WorkMode.CLUSTER and backend == Backend.EGGROLL:
            from fate_arch.storage import EggRollStorageType
            address_dict = {"name": table_name, "namespace": namespace,
                            "storage_type": EggRollStorageType.ROLLPAIR_LMDB}
            storage_engine = StorageEngine.EGGROLL
        elif work_mode == WorkMode.CLUSTER and backend == Backend.SPARK:
            address_dict = {"path": data_utils.default_output_fs_path(name=table_name, namespace=namespace,
                                                                      prefix=path_prefix)}
            storage_engine = StorageEngine.HDFS
        else:
            raise RuntimeError(f"Unknown work_mode: {work_mode} or backend {backend} given.")
        address = storage.StorageTableMeta.create_address(storage_engine=storage_engine, address_dict=address_dict)
        table.save(address, schema=table.schema, partitions=table.partitions)

        part_of_data = []
        part_of_limit = 100
        for k, v in table.collect():
            part_of_data.append((k, v))
            part_of_limit -= 1
            if part_of_limit == 0:
                break
        table_count = table.count()
        save_schema = copy.deepcopy(table.schema)
        table_meta = storage.StorageTableMeta(name=table_name, namespace=namespace, new=True)
        table_meta.address = address
        table_meta.partitions = table.partitions
        table_meta.engine = storage_engine
        table_meta.type = storage.EggRollStorageType.ROLLPAIR_LMDB
        table_meta.schema = save_schema
        table_meta.part_of_data = part_of_data
        table_meta.count = table_count
        table_meta.create()
        session.computing.kill()

    def run(p):
        while p.get_switch():
            time.sleep(1)
            p.progress(p.get_time_percent())

    if not match_rate > 0 or not match_rate <= 1:
        raise Exception(f"The value is between (0-1), Please check match_rate:{match_rate}")
    guest_start_num = host_data_size - int(guest_data_size * match_rate)
    guest_end_num = guest_start_num + guest_data_size

    if os.path.isfile(include_path):
        with include_path.open("r") as f:
            testsuite_config = json.load(f)
    else:
        raise Exception(f'Input file error, please check{include_path}.')
    try:
        if output_path is not None:
            big_data_dir = os.path.abspath(output_path)
        else:
            big_data_dir = os.path.abspath(conf.cache_directory)
    except Exception:
        raise Exception('{}path does not exist'.format(big_data_dir))
    date_set = {}
    table_name_list = []
    table_namespace_list = []
    for upload_dict in testsuite_config.get('data'):
        date_set[os.path.basename(upload_dict.get('file'))] = upload_dict.get('role')
        table_name_list.append(upload_dict.get('name'))
        table_namespace_list.append(upload_dict.get('namespace'))
    data_count = 0
    for idx, data_name in enumerate(date_set.keys()):
        label_flag = True if 'guest' in date_set[data_name] else False
        data_type = 'dense' if 'guest' in date_set[data_name] else host_data_type
        if split_host and ('host' in date_set[data_name]):
            host_end_num = int(np.ceil(host_data_size / len(date_set))) * (data_count + 1) if np.ceil(
                host_data_size / len(date_set)) * (data_count + 1) <= host_data_size else host_data_size
            host_start_num = int(np.ceil(host_data_size / len(date_set))) * data_count
            data_count += 1
        else:
            host_end_num = host_data_size
            host_start_num = 0
        out_path = os.path.join(str(big_data_dir), data_name)
        if os.path.exists(out_path) and os.path.isfile(out_path):
            if force:
                remove_file(out_path)
            else:
                raise Exception('{} Already exists'.format(out_path))
        data_i = (idx + 1) / len(date_set)
        downLoad = f'dataget  [{"#" * int(24 * data_i)}{"-" * (24 - int(24 * data_i))}]  {idx + 1}/{len(date_set)}'
        start = time.time()
        progress = data_progress(downLoad, start)
        thread = threading.Thread(target=run, args=[progress])
        thread.start()
        work_mode = conf.work_mode
        backend = conf.backend
        try:
            if 'guest' in date_set[data_name]:
                if not parallelize:
                    _generate_dens_data(out_path, guest_start_num, guest_end_num, guest_feature_num, label_flag)
                else:
                    _generate_parallelize_data(guest_start_num, guest_end_num, guest_feature_num,
                                               table_name_list[idx], table_namespace_list[idx], path_prefix="")
            else:
                if data_type == 'tag' and not parallelize:
                    _generate_tag_data(out_path, host_start_num, host_end_num, host_feature_num, sparsity)
                elif data_type == 'tag_value' and not parallelize:
                    _generate_tag_value_data(out_path, host_start_num, host_end_num, host_feature_num)
                elif data_type == 'dense' and not parallelize:
                    _generate_dens_data(out_path, host_start_num, host_end_num, host_feature_num, label_flag)
                elif parallelize:
                    _generate_parallelize_data(host_start_num, host_end_num, guest_feature_num,
                                               table_name_list[idx], table_namespace_list[idx], path_prefix="")
            progress.set_switch(False)
            time.sleep(1)
            print()
        except Exception:
            progress.set_switch(False)
            raise Exception(f"Output file failed")
