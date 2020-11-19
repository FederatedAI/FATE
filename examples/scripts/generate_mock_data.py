import random
import sys

SAMPLE_NUM = 10000

# For sparse data, it means how many non-zero features in one sample.
# The total possible feature num depends on your tag interval below.
FEATURE_NUM = 20

# Should be one of "tag_integer_value", "tag", "tag_float_value", "tag_1" or "label"
# DATA_TYPE = "tag_integer_value"

# TAG_INTERVAL = (1, 5000)
# TAG_INTERVAL = (2019010101, 2019011101)
TAG_INTERVAL = (2019120799, 2019121299)
# SAVE_FILE_NAME = DATA_TYPE + "_" + str(SAMPLE_NUM) + "_" + str(FEATURE_NUM) + ".csv"

VALUE_LENGTH = 4
VALUE_INTERVAL = (1, 10)


def generate_tag_1_data(ids):
    if len(ids) != SAMPLE_NUM:
        raise ValueError("len ids should equal to sample number")

    counter = 0
    for sample_i in range(SAMPLE_NUM):
        one_data = [ids[sample_i]]
        for feature_i in range(FEATURE_NUM):
            tag = str(random.randint(TAG_INTERVAL[0], TAG_INTERVAL[1]))
            value = '1.0'
            tag_value = ":".join([tag, value])
            one_data.append(tag_value)

        counter += 1
        if counter % 10000 == 0:
            print("generate data {}".format(counter))

        yield one_data


def generate_tag_float_value_data(ids):
    if len(ids) != SAMPLE_NUM:
        raise ValueError("len ids should equal to sample number")

    counter = 0
    for sample_i in range(SAMPLE_NUM):
        one_data = [ids[sample_i]]
        for feature_i in range(FEATURE_NUM):
            tag = str(random.randint(TAG_INTERVAL[0], TAG_INTERVAL[1]))
            value = str(round(100 * random.random(), 2))
            tag_value = ":".join([tag, value])
            one_data.append(tag_value)

        counter += 1
        if counter % 10000 == 0:
            print("generate data {}".format(counter))

        yield one_data


def generate_tag_data(ids):
    if len(ids) != SAMPLE_NUM:
        raise ValueError("len ids should equal to sample number")

    counter = 0
    # v_str = "0123456789abcd"
    for sample_i in range(SAMPLE_NUM):
        one_data = [ids[sample_i]]
        for feature_i in range(FEATURE_NUM):
            tag = str(random.randint(TAG_INTERVAL[0], TAG_INTERVAL[1]))
            # value = ''
            # for i in range(VALUE_LENGTH):
            #     value += v_str[int(random.random() * 14)]
            # tag_value = ":".join([tag, value])
            one_data.append(tag)

        counter += 1
        if counter % 10000 == 0:
            print("generate data {}".format(counter))

        yield one_data


def generate_tag_value_data(ids):
    if len(ids) != SAMPLE_NUM:
        print(len(ids))
        print(SAMPLE_NUM)
        raise ValueError("len ids should equal to sample number")

    counter = 0
    for sample_i in range(SAMPLE_NUM):
        one_data = [ids[sample_i]]
        for feature_i in range(FEATURE_NUM):
            tag = str(random.randint(TAG_INTERVAL[0], TAG_INTERVAL[1]))
            value = str(random.randint(VALUE_INTERVAL[0], VALUE_INTERVAL[1]))
            tag_value = ":".join([tag, value])
            one_data.append(tag_value)

        counter += 1
        if counter % 10000 == 0:
            print("generate data {}".format(counter))

        yield one_data


def generate_label_data(ids):
    if len(ids) != SAMPLE_NUM:
        print(len(ids))
        print(SAMPLE_NUM)
        raise ValueError("len ids should equal to sample number")

    header = ['id', 'y'] + ['x' + str(i) for i in range(FEATURE_NUM)]
    yield header

    counter = 0
    for sample_i in range(SAMPLE_NUM):
        one_data = [ids[sample_i], round(random.random())]
        for feature_i in range(FEATURE_NUM):
            one_data.append(random.random())
        counter += 1
        if counter % 10000 == 0:
            print("generate data {}".format(counter))

        yield one_data


def read_file(file, has_header=False):
    header = None
    data = []
    with open(file, "r") as fin:
        if has_header:
            header = fin.readline().replace('\n', '')
        line = fin.readline()

        while True:
            split_line = line.replace("\n", '').split(",")
            data.append(split_line)

            line = fin.readline()
            if not line:
                break

    return header, data


def save_file(file, data, header=None, delimitor=','):
    with open(file, 'w') as fout:
        if header:
            fout.write("".join([header, '\n']))

        for d in data:
            d = list(map(str, d))
            fout.write(d[0] + ',' + delimitor.join(d[1:]) + "\n")


if __name__ == "__main__":
    # ids = [_data[0] for _data in ids_data]
    DATA_TYPE = sys.argv[1]
    role = sys.argv[2]
    SAVE_FILE_NAME = "generated_data_{}.csv".format(role)

    ids = [x for x in range(SAMPLE_NUM)]

    if DATA_TYPE == 'tag_1':
        new_data = generate_tag_1_data(ids)
        save_file(SAVE_FILE_NAME, new_data, delimitor=',')

    if DATA_TYPE == 'tag_float_value':
        new_data = generate_tag_float_value_data(ids)
        save_file(SAVE_FILE_NAME, new_data, delimitor=';')

    if DATA_TYPE == 'tag':
        new_data = generate_tag_data(ids)
        save_file(SAVE_FILE_NAME, new_data, delimitor=';')

    if DATA_TYPE == 'tag_integer_value':
        new_data = generate_tag_value_data(ids)
        save_file(SAVE_FILE_NAME, new_data, delimitor=',')

    if DATA_TYPE == 'label':
        new_data = generate_label_data(ids)
        save_file(SAVE_FILE_NAME, new_data, delimitor=',')

    print("finish generate data , save data in {}".format(SAVE_FILE_NAME))
