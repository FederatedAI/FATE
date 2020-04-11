import sys
from copy import deepcopy

settings_dict = {
    "CHECK_NODES_IDENTITY": "CHECK_NODES_IDENTITY = False",
     "MANAGER_HOST": "MANAGER_HOST = server_conf.get(SERVERS).get('fatemanager', {}).get('host')",
     "MANAGER_PORT": "MANAGER_PORT = server_conf.get(SERVERS).get('fatemanager', {}).get('port')" ,
     "JOB_DEFAULT_TIMEOUT": "JOB_DEFAULT_TIMEOUT = 7 * 24 * 60 * 60",
     "FATE_MANAGER_GET_NODE_INFO": "FATE_MANAGER_GET_NODE_INFO = '/node/info'",
     "FATE_MANAGER_NODE_CHECK": "FATE_MANAGER_NODE_CHECK = '/node/management/check'",
     "FATE_FLOW_MODEL_TRANSFER_PATH": "FATE_FLOW_MODEL_TRANSFER_PATH = '/v1/model/transfer'",
     "SERVING_PATH": "SERVING_PATH = '/servers/servings'",
     "SERVER_CONF_PATH": "SERVER_CONF_PATH = 'arch/conf/server_conf.json'"
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
