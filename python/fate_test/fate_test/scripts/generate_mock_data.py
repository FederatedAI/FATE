import hashlib
import json
import os
import random
import threading
import sys
import time
import pandas as pd
import numpy as np
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
    if os.path.exists(path) and os.path.isfile(path):
        os.remove(path)


def get_big_data(task, guest_data_size, host_data_size, guest_feature_num, host_feature_num, include_path, conf: Config,
                 encryption_type, match_rate, sparsity):
    def list_tag_value(feature_nums):
        data = ''
        for f in range(feature_nums):
            data += ('x' + str(f)) + ':' + str(round(np.random.random(), 2)) + ";"
        return data[:-1]

    def list_tag(feature_nums, data_list):
        data = ''
        for f in range(feature_nums):
            data += random.choice(data_list) + ";"
        return data[:-1]

    def _generate_tag_value_data(data_path, data_num, id_value, feature_nums):
        section_data_size = round(data_num / 100)
        iteration = round(data_num / section_data_size)
        for batch in range(iteration + 1):
            progress.set_time_percent(batch)
            output_data = pd.DataFrame(columns=["id"])
            if section_data_size * (batch + 1) <= data_num:
                output_data['id'] = id_value[section_data_size * batch: section_data_size * (batch + 1)]
                data_size = section_data_size
            elif section_data_size * batch <= data_num:
                output_data['id'] = id_value[section_data_size * batch: data_num]
                data_size = data_num - section_data_size * batch
            else:
                break
            feature = [list_tag_value(feature_nums) for i in range(data_size)]
            output_data['feature'] = feature
            output_data.to_csv(data_path, mode='a+', index=False, header=False)

    def _generate_label_data(data_path, data_num, id_value, feature_nums):
        head_1 = ['id', 'y']
        head_2 = ['x' + str(i) for i in range(feature_nums)]
        df_data_1 = pd.DataFrame(columns=head_1)
        head_data = pd.DataFrame(columns=head_1 + head_2)
        head_data.to_csv(data_path, mode='a+', index=False)
        section_data_size = round(data_num / 100)
        iteration = round(data_num / section_data_size)
        for batch in range(iteration + 1):
            progress.set_time_percent(batch)
            if section_data_size * (batch + 1) <= data_num:
                df_data_1["id"] = id_value[section_data_size * batch: section_data_size * (batch + 1)]
                df_data_1["y"] = [round(np.random.random()) for x in range(section_data_size)]
                data_size = section_data_size
            elif section_data_size * batch < data_num:
                df_data_1["id"] = id_value[section_data_size * batch: data_num]
                data_size = data_num - section_data_size * batch
                df_data_1["y"] = [round(np.random.random()) for x in range(data_size)]
            else:
                break
            feature = np.random.randint(0, 100, size=[data_size, feature_nums]) / 100
            df_data_2 = pd.DataFrame(feature, columns=head_2)
            output_data = pd.concat([df_data_1, df_data_2], axis=1)
            output_data.to_csv(data_path, mode='a+', index=False, header=False)

    def _generate_tag_data(data_path, data_num, id_value, feature_nums, sparsity):
        section_data_size = round(data_num / 100)
        iteration = round(data_num / section_data_size)
        valid_set = [x for x in range(2019120799, 2019120799 + round(feature_nums / sparsity))]
        data = list(map(str, valid_set))
        for batch in range(iteration + 1):
            progress.set_time_percent(batch)
            output_data = pd.DataFrame()
            if section_data_size * (batch + 1) <= data_num:
                output_data[0] = id_value[section_data_size * batch: section_data_size * (batch + 1)]
                data_size = section_data_size
            elif section_data_size * batch <= data_num:
                output_data[0] = id_value[section_data_size * batch: data_num]
                data_size = data_num - section_data_size * batch
            else:
                break
            feature = [list_tag(feature_nums, data_list=data) for i in range(data_size)]
            output_data['feature'] = feature
            output_data.to_csv(data_path, mode='a+', index=False, header=False)

    def run(p):
        while p.get_switch():
            time.sleep(1)
            p.progress(p.get_time_percent())

    if not match_rate > 0 or not match_rate <= 1:
        raise Exception(f"The value is between (0-1), Please check match_rate:{match_rate}")
    intersect_count = int(guest_data_size * match_rate)
    non_intersect_count = guest_data_size - intersect_count

    guest_ids = [str(x) for x in range(intersect_count)]
    non_intersect_ids = [str(x) for x in range(host_data_size, host_data_size + non_intersect_count)]
    guest_ids = guest_ids + non_intersect_ids
    host_ids = [str(x) for x in range(host_data_size)]
    if encryption_type == 'md5':
        guest_ids = [hashlib.md5(bytes(value, encoding='utf-8')).hexdigest() for value in guest_ids]
        host_ids = [hashlib.md5(bytes(value, encoding='utf-8')).hexdigest() for value in host_ids]
    elif encryption_type == 'sha256':
        guest_ids = [hashlib.sha256(bytes(value, encoding='utf-8')).hexdigest() for value in guest_ids]
        host_ids = [hashlib.sha256(bytes(value, encoding='utf-8')).hexdigest() for value in host_ids]

    if os.path.isfile(include_path):
        with include_path.open("r") as f:
            testsuite_config = json.load(f)
    else:
        raise Exception(f'Input file error, please check{include_path}.')
    big_data_dir = conf.cache_directory
    if not os.path.isdir(big_data_dir):
        os.mkdir(big_data_dir)
    date_set = [os.path.basename(upload_dict['file']) for upload_dict in testsuite_config['data']]
    data_count = 0
    for idx, data_name in enumerate(date_set):
        if task == 'intersect_multi' and ('guest' in data_name or 'label' in data_name):
            right = int(np.ceil(guest_data_size / len(date_set))) * (data_count + 1) if np.ceil(
                guest_data_size / len(date_set)) * (data_count + 1) <= guest_data_size else guest_data_size
            guest_id_list = guest_ids[int(np.ceil(guest_data_size / len(date_set))) * data_count: right]
            data_count += 1
        else:
            guest_id_list = guest_ids
        out_path = os.path.join(str(big_data_dir), data_name)
        remove_file(out_path)
        data_i = (idx + 1) / len(date_set)
        downLoad = f'dataget  [{"#" * int(24 * data_i)}{"-" * (24 - int(24 * data_i))}]  {idx + 1}/{len(date_set)}'
        start = time.time()
        progress = data_progress(downLoad, start)
        thread = threading.Thread(target=run, args=[progress])
        thread.start()
        try:
            if 'tag' in data_name and 'tag_value' not in data_name:
                _generate_tag_data(out_path, host_data_size, host_ids, host_feature_num, sparsity)
            elif 'tag_value' in data_name:
                _generate_tag_value_data(out_path, host_data_size, host_ids, host_feature_num)
            elif 'guest' in data_name or 'label' in data_name:
                _generate_label_data(out_path, guest_data_size, guest_id_list, guest_feature_num)
            else:
                progress.set_switch(False)
                raise Exception(
                    f'The host file name contains "tag" or "tag_value", and the guest file name contains "guest" or "label". Please check your file name: {data_name}')
            progress.set_switch(False)
            time.sleep(1)
            print()
        except Exception:
            progress.set_switch(False)
            raise Exception(f"Output file failed")
