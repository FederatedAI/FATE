import numpy as np
import random
import time
import json
from model.model_wrapper import Models
from socketIO_client import SocketIO
from utils.model_dump import *

import os

import logging
import argparse

logging.getLogger('socketIO-client').setLevel(logging.WARNING)
random.seed(2018)
datestr = time.strftime('%m%d')
log_dir = os.path.join('experiments', 'logs', datestr)
if not os.path.exists(log_dir):
    raise FileNotFoundError("{} not found".format(log_dir))


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


class LocalModel(object):
    def __init__(self, task_config):
        """
        Inputs:
            model: should be a python class refering to pytorch model (torch.nn.Module)
            data_collected: a list with train/val/test dataset
        """
        self.model_name = task_config['model_name']
        self.epoch = task_config['local_epoch']
        self.model = getattr(Models, self.model_name)(task_config)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    def train_one_round(self):
        losses = []
        for i in range(1, self.epoch + 1):
            loss = self.model.train_one_epoch()
            losses.append(loss)
        # total_loss, mAP, recall = self.model.eval(self.model.dataloader, self.model.yolo, test_num=1000)
        #return self.model.get_weights(), total_loss, mAP, recall
        return self.model.get_weights(), sum(losses) / len(losses)

    def evaluate(self):
        loss, acc, recall = self.model.evaluate()
        return loss, acc, recall


# A federated client is a process that can go to sleep / wake up intermittently
# it learns the global model by communication with the server;
# it contributes to the global model by sending its local gradients.

class FederatedClient(object):
    MAX_DATASET_SIZE_KEPT = 6000

    def __init__(self, server_host, server_port, task_config_filename,
                 gpu, ignore_load):
        os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
        self.task_config = load_json(task_config_filename)
        # self.data_path = self.task_config['data_path']
        print(self.task_config)
        self.ignore_load = ignore_load

        self.local_model = None
        self.dataset = None

        self.log_filename = self.task_config['log_filename']
        # logger
        self.logger = logging.getLogger("client")
        self.fh = logging.FileHandler(os.path.join(log_dir, os.path.basename(self.log_filename)))
        self.fh.setLevel(logging.INFO)
        # create console handler with a higher log level
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(self.formatter)
        self.ch.setFormatter(self.formatter)
        # add the handlers to the logger
        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)
        self.logger.info(self.task_config)
        self.sio = SocketIO(server_host, server_port, None, {'timeout': 36000})
        self.register_handles()
        print("sent wakeup")
        self.sio.emit('client_wake_up')
        self.sio.wait()

    ########## Socket Event Handler ##########
    def on_init(self, request):
        print('on init')
        self.local_model = LocalModel(self.task_config)
        print("local model initialized done.")
        # ready to be dispatched for training
        self.sio.emit('client_ready')

    def load_stat(self):
        loadavg = {}
        with open("/proc/loadavg") as fin:
            con = fin.read().split()
            loadavg['lavg_1'] = con[0]
            loadavg['lavg_5'] = con[1]
            loadavg['lavg_15'] = con[2]
            loadavg['nr'] = con[3]
            loadavg['last_pid'] = con[4]
        return loadavg['lavg_15']

    def register_handles(self):
        ########## Socket IO messaging ##########
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_request_update(*args):

            req = args[0]
            print("update requested")

            cur_round = req['round_number']
            self.logger.info("### Round {} ###".format(cur_round))

            if cur_round == 0:
                self.logger.info("received initial model")
                print(req['current_weights'])
                weights = pickle_string_to_obj(req['current_weights'])
                self.local_model.set_weights(weights)

            my_weights, train_loss = self.local_model.train_one_round()
            print(train_loss)

            pickle_string_weights = obj_to_pickle_string(my_weights)
            resp = {
                'round_number': cur_round,
                'weights': pickle_string_weights,
                'train_size': self.local_model.model.train_size,
                'train_loss': train_loss
            }

            self.logger.info("client_train_loss {}".format(train_loss))

            if 'aggregation' in req and req['aggregation']:
                client_test_loss, client_test_map, client_test_recall = self.local_model.evaluate()
                client_test_map = np.nan_to_num(client_test_map)
                client_test_recall = np.nan_to_num(client_test_recall)
                resp['client_test_loss'] = client_test_loss
                resp['client_test_map'] = client_test_map
                resp['client_test_recall'] = client_test_recall
                resp['client_test_size'] = self.local_model.model.valid_size
                self.logger.info("client_test_loss {}".format(client_test_loss))
                self.logger.info("client_test_map {}".format(client_test_map))
                self.logger.info("client_test_recall {}".format(client_test_recall))

            print("Emit client_update")
            self.sio.emit('client_update', resp)
            self.logger.info("sent trained model to server")
            print("Emited...")

        def on_stop_and_eval(*args):
            self.logger.info("received aggregated model from server")
            req = args[0]
            cur_time = time.time()
            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])
                self.local_model.set_weights(weights)
            print('get weights')

            self.logger.info("reciving weight time is {}".format(time.time() - cur_time))
            server_loss, server_map, server_recall = self.local_model.evaluate()
            server_map = np.nan_to_num(server_map)
            server_recall = np.nan_to_num(server_recall)
            resp = {
                'test_size': self.local_model.model.valid_size,
                'test_loss': server_loss,
                'test_map': server_map,
                'test_recall': server_recall
            }
            print("Emit client_eval")
            self.sio.emit('client_eval', resp)

            if req['STOP']:
                print("Federated training finished ...")
                exit(0)

        def on_check_client_resource(*args):
            req = args[0]
            print("check client resource.")
            if self.ignore_load:
                load_average = 0.15
                print("Ignore load average")
            else:
                load_average = self.load_stat()
                print("Load average:", load_average)

            resp = {
                'round_number': req['round_number'],
                'load_rate': load_average
            }
            self.sio.emit('check_client_resource_done', resp)

        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', self.on_init)
        self.sio.on('request_update', on_request_update)
        self.sio.on('stop_and_eval', on_stop_and_eval)
        self.sio.on('check_client_resource', on_check_client_resource)

        # TODO: later: simulate datagen for long-running train-serve service
        # i.e. the local dataset can increase while training

        # self.lock = threading.Lock()
        # def simulate_data_gen(self):
        #     num_items = random.randint(10, FederatedClient.MAX_DATASET_SIZE_KEPT * 2)
        #     for _ in range(num_items):
        #         with self.lock:
        #             # (X, Y)
        #             self.collected_data_train += [self.datasource.sample_single_non_iid()]
        #             # throw away older data if size > MAX_DATASET_SIZE_KEPT
        #             self.collected_data_train = self.collected_data_train[-FederatedClient.MAX_DATASET_SIZE_KEPT:]
        #             print(self.collected_data_train[-1][1])
        #         self.intermittently_sleep(p=.2, low=1, high=3)

        # threading.Thread(target=simulate_data_gen, args=(self,)).start()

    def intermittently_sleep(self, p=.1, low=10, high=100):
        if (random.random() < p):
            time.sleep(random.randint(low, high))


# possible: use a low-latency pubsub system for gradient update, and do "gossip"
# e.g. Google cloud pubsub, Amazon SNS
# https://developers.google.com/nearby/connections/overview
# https://pypi.python.org/pypi/pyp2p

# class PeerToPeerClient(FederatedClient):
#     def __init__(self):
#         super(PushBasedClient, self).__init__()    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="which GPU to run")
    parser.add_argument("--config_file", type=str, required=True, help="task config file")
    parser.add_argument("--ignore_load", default=True, help="wheter ignore load of not")
    parser.add_argument("--port", type=int, required=True, help="server port")
    opt = parser.parse_args()
    print(opt)
    if not os.path.exists(opt.config_file):
        raise FileNotFoundError('{} does not exist'.format(opt.config_file))
    print("client run on {}".format(opt.gpu))
    try:
        FederatedClient("127.0.0.1", opt.port, opt.config_file, opt.gpu, opt.ignore_load)
    except ConnectionError:
        print('The server is down. Try again later.')
