import logging
import argparse
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
import syft as sy
from syft.workers import websocket_server
from torch.autograd import Variable
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import dataloader_detector

global config
config = {}
# config['anchors'] = [ 10.0, 30.0, 60.]
config['anchors'] = [30.0]
config['chanel'] = 1
config['crop_size'] = [128, 128, 128]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 1  ## 原来是2，每个图里选两个negative
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 6. #mm
config['sizelim2'] = 30
config['sizelim3'] = 40
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','990fbe3f0a1b53878669967b9afd1441','adc3bbc63d40f8761c59be10f1e504c3']

print('-------------------------------------------------------')
print('=========================INPUT:', config)
print('-------------------------------------------------------')


def start_websocket_server_worker(id, host, port, hook, verbose):
    """Helper function for spinning up a websocket server and setting up the local datasets."""

    server = websocket_server.WebsocketServerWorker(id=id, host=host, port=port, hook=hook, verbose=verbose)

    dataset_train = DataLoader(dataloader_detector.get_trainloader("train", config, args),
        batch_size = args.batch_size,
        shuffle = True,
        pin_memory = False,
        )

    data_total   = []
    target_total = []
    coord_total  = []

    for i, (data, target, coord) in enumerate(dataset_train):
        data_total.append(data.numpy().reshape([1, 128, 128, 128]))
        #target_total.append(target)
        #data_total.append(np.concatenate((data.numpy().reshape(1, 1*1*128*128*128), coord.numpy().reshape(1, 1*3*32*32*32)), axis = 1))
        target_total.append(target.numpy().reshape([32, 32, 32, 1, 5]))
        coord_total.append(coord.numpy().reshape([3, 32, 32, 32]))


    dataset = sy.BaseDataset_zhiyuan(data_1=data_total, data_2=coord_total, targets=target_total)
    key = "testing"

    server.add_dataset(dataset, key=key)
    logger.info("Testing dataset (%s set), available numbers on %s: ", "train", id)
    logger.info("Datasets: %s", server.datasets)
    logger.info("lenth of LUNA2016 Datasets: %s", len(server.datasets[key]))

    server.start()

    return server


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("run_websocket_server")
    logger.setLevel(level=logging.DEBUG)

    # Parse args
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument("--port", "-p", type=int,
        help="port number of the websocket server worker, e.g. --port 8777")
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    parser.add_argument("--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice")
    parser.add_argument("--verbose", "-v", action="store_true", 
        help="if set, websocket server worker will be started in verbose mode")
    parser.add_argument('--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--save-dir', default='results', type=str, metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    parser.add_argument('--test', default=0, type=int, metavar='TEST',
                        help='1 do test evaluation, 0 not')
    parser.add_argument('--split', default=1, type=int, metavar='SPLIT',
                        help='In the test phase, split the image to 8 parts')
    parser.add_argument('--validation-subset', default=0, type=int, metavar='N',
                        help='choose a subset [0~9] for validation, the rest will be for training')
    parser.add_argument('--training-subset', default=0, type=int, metavar='N',
                        help='choose a subset [0~9] for training, 100 is using all except validation subset')
    #global args
    args = parser.parse_args()


    # Hook and start server
    hook = sy.TorchHook(torch)
    server = start_websocket_server_worker(
        id=args.id,
        host=args.host,
        port=args.port,
        hook=hook,
        verbose=args.verbose,
        )