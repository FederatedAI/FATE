import sys
from copy import deepcopy

settings_dict = {
    "USE_LOCAL_DATABASE": "USE_LOCAL_DATABASE = True"
                 }



def modify_file(file_name):
    settings_list = list(settings_dict.keys())
    cp = deepcopy(settings_list)
    with open(file_name) as fp:
        lines = fp.readlines()
        for line in lines:
            for key in list(set(settings_list)):
                if key in line:
                    try:
                        cp.remove(key)
                    except:
                        pass

    with open(file_name, 'a') as f:
        f.write('\n')
        for key in cp:
            f.write(settings_dict[key]+'\n')


if __name__ == '__main__':
    file_name = sys.argv[1]
    modify_file(file_name)
