import random
import json
import os
import hashlib
import numpy as np
from fate_test._config import Config


def remove_file(path):
    if os.path.exists(path) and os.path.isfile(path):
        os.remove(path)


def get_big_data(task, guest_data_size, host_data_size, guest_feature_num, host_feature_num, include_path, conf: Config,
                 encryption_type, match_rate):
    def _generate_tag_float_value_data(id_value, feature_nums):
        counter = 0
        for sample_i in range(host_data_size):
            one_data = [id_value[sample_i]]
            valid_set = ['x' + str(i) for i in range(feature_nums)]
            tags = np.random.choice(valid_set, feature_nums, replace=False)
            values = np.random.random(feature_nums)
            one_data += [":".join([str(tags[i]), str(round(values[i], 2))]) for i in range(feature_nums)]
            counter += 1
            yield one_data

    def _generate_tag_data(id_value, feature_nums):
        counter = 0
        for sample_i in range(host_data_size):
            one_data = [id_value[sample_i]]
            valid_set = [x for x in range(2019120799, 2019121299)]
            tags = np.random.choice(valid_set, feature_nums, replace=False)
            one_data += [str(tag) for tag in tags]

            counter += 1
            if counter % 10000 == 0:
                print("generate data {}".format(counter))

            yield one_data

    def _generate_label_data(id_value, feature_nums):
        header = ['id', 'y'] + ['x' + str(i) for i in range(feature_nums)]
        yield header
        counter = 0
        for sample_i in range(len(id_value)):
            one_data = [id_value[sample_i], round(random.random())] + list(np.random.random(feature_nums))
            counter += 1
            yield one_data

    def _save_file(file, data, header=None, delimitor=','):
        try:
            if not os.path.exists(file):
                with open(file, 'w') as f_out:
                    if header:
                        f_out.write("".join([header, '\n']))

                    for d in data:
                        d = list(map(str, d))
                        f_out.write(d[0] + ',' + delimitor.join(d[1:]) + "\n")
            else:
                raise Exception(f"His file already exists.")
        except Exception as e:
            raise Exception(f"write error==>{e}")

    if not match_rate:
        intersect_count = guest_data_size
        non_intersect_count = 0
    else:
        if not match_rate > 0 or not match_rate < 1:
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
    for data_name in date_set:
        if task == 'intersect_multi' and ('guest' in data_name or 'label' in data_name):
            right = np.ceil(guest_data_size / len(date_set)) * (data_count + 1) if np.ceil(
                guest_data_size / len(date_set)) * (data_count + 1) <= guest_data_size else guest_data_size
            guest_id_list = guest_ids[np.ceil(guest_data_size / len(date_set)) * data_count: right]
            data_count += 1
        else:
            guest_id_list = guest_ids
        data_path = os.path.join(str(big_data_dir), data_name)
        remove_file(data_path)
        try:
            if 'tag' in data_name and 'tag_value' not in data_name:
                _save_file(data_path, _generate_tag_data(host_ids, host_feature_num), delimitor=',')
            elif 'tag_value' in data_name:
                _save_file(data_path, _generate_tag_float_value_data(host_ids, host_feature_num), delimitor=';')
            elif 'guest' in data_name or 'label' in data_name:
                _save_file(data_path, _generate_label_data(guest_id_list, guest_feature_num), delimitor=',')
            else:
                raise Exception(
                    f'The host file name contains "tag" or "tag_value", and the guest file name contains "guest" or "label". Please check your file name: {data_name}')
        except Exception as e:
            raise Exception(f"Output file failed: {e}")
