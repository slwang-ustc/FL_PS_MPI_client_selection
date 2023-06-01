import copy
import asyncio
from typing import List

from torch import nn

from comm_utils import send_data, get_data
from config import cfg
import os
import time
import random
from random import sample

import numpy as np
import torch
from client import *
import datasets
from models import utils
from training_utils import test

from mpi4py import MPI

import logging

random.seed(cfg['client_selection_seed'])

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['server_cuda']
device = torch.device("cuda" if cfg['server_use_cuda'] and torch.cuda.is_available() else "cpu")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

RESULT_PATH = os.getcwd() + '/server_log/'
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)
# init logger
logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
filename = RESULT_PATH + now + "_" + os.path.basename(__file__).split('.')[0] + '.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

comm_tags = np.ones(cfg['client_num'] + 1)


def main():
    client_num = cfg['client_num']
    logger.info("Total number of clients: {}".format(client_num))
    logger.info("\nModel type: {}".format(cfg["model_type"]))
    logger.info("Dataset: {}".format(cfg["dataset_type"]))

    # init the global model
    global_model = utils.create_model_instance(cfg['model_type'])
    global_model.to(device)
    global_params = nn.utils.parameters_to_vector(global_model.parameters())
    global_params_num = global_params.nelement()
    global_model_size = global_params_num * 4 / 1024 / 1024
    logger.info("Global params num: {}".format(global_params_num))
    logger.info("Global model Size: {} MB".format(global_model_size))

    # partition the dataset
    train_data_partition, partition_sizes = partition_data(
        dataset_type=cfg['dataset_type'],
        partition_pattern=cfg['data_partition_pattern'],
        non_iid_ratio=cfg['non_iid_ratio'],
        client_num=client_num
    )

    logger.info('\nData partition: ')
    for i in range(len(partition_sizes)):
        s = ""
        for j in range(len(partition_sizes[i])):
            s += "{:.2f}".format(partition_sizes[i][j]) + " "
        logger.info(s)

    # create clients
    all_clients: List[ClientConfig] = list()
    for client_idx in range(client_num):
        client = ClientConfig(client_idx)
        client.lr = cfg['lr']
        all_clients.append(client)

    # load the test dataset and test loader
    _, test_dataset = datasets.load_datasets(cfg['dataset_type'], cfg['dataset_path'])
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=cfg['test_batch_size'], shuffle=False)

    # begin each epoch
    for epoch_idx in range(1, 1 + cfg['epoch_num']):
        logger.info("_____****_____\nEpoch: {:04d}".format(epoch_idx))
        print("_____****_____\nEpoch: {:04d}".format(epoch_idx))

        # The client selection algorithm can be implemented
        selected_num = 10
        selected_client_idxes = sample(range(client_num), selected_num)
        logger.info("Selected clients' idxes: {}".format(selected_client_idxes))
        print("Selected clients' idxes: {}".format(selected_client_idxes))

        # create instances of the selected clients
        selected_clients = []
        for client_idx in selected_client_idxes:
            client = ClientConfig(idx=client_idx)
            for k, v in all_clients[client_idx].__dict__.items():
                setattr(client, k, v)
            client.epoch_idx = epoch_idx
            client.params_dict = global_model.state_dict()
            client.train_data_idxes = train_data_partition.use(client_idx)
            selected_clients.append(client)

        # send the configurations to the selected clients
        communication_parallel(selected_clients, comm_tags, comm, action="send_config")

        # when all selected clients have completed local training, receive their configurations
        communication_parallel(selected_clients, comm_tags, comm, action="get_config")

        for client in selected_clients:
            for k, v in client.__dict__.items():
                if k != 'params_dict' and k != 'train_data_idxes':
                    setattr(all_clients[client.idx], k, v)

        # aggregate the clients' local model parameters
        aggregate_models(global_model, selected_clients)

        # test the global model
        test_loss, test_acc = test(global_model, test_loader, device)
        logger.info(
            "Test_Loss: {:.4f}\n".format(test_loss) +
            "Test_ACC: {:.4f}\n".format(test_acc)
        )

        for m in range(len(selected_clients)):
            comm_tags[m + 1] += 1


def aggregate_models(global_model, client_list):
    with torch.no_grad():
        params_dict = copy.deepcopy(global_model.state_dict())
        for client in client_list:
            for k, v in client.params_dict.items():
                if 'num_batches_tracked' not in k: 
                    params_dict[k] += \
                        client.aggregate_weight * (client.params_dict[k] - params_dict[k])
    global_model.load_state_dict(params_dict)


async def send_config(client, rank, comm, comm_tag):
    await send_data(comm, client, rank, comm_tag)


async def get_config(client, comm, rank, comm_tag):
    config_received = await get_data(comm, rank, comm_tag)
    for k, v in config_received.__dict__.items():
        setattr(client, k, v)


def communication_parallel(client_list, comm_tags, comm, action):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for m, client in enumerate(client_list): 
        if action == "send_config":
            task = asyncio.ensure_future(
                send_config(client, m + 1, comm, comm_tags[m + 1])
            )
        elif action == "get_config":
            task = asyncio.ensure_future(
                get_config(client, comm, m + 1, comm_tags[m + 1])
            )
        else:
            raise ValueError('Not valid action')
        tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


def partition_data(dataset_type, partition_pattern, non_iid_ratio, client_num=10):
    train_dataset, _ = datasets.load_datasets(dataset_type=dataset_type, data_path=cfg['dataset_path'])
    partition_sizes = np.ones((cfg['classes_size'], client_num))
    # iid
    if partition_pattern == 0:
        partition_sizes *= (1.0 / client_num)
    # non-iid
    # each client contains all classes of data, but the proportion of certain classes of data is very large
    elif partition_pattern == 1:
        if 0 < non_iid_ratio < 10:
            partition_sizes *= ((1 - non_iid_ratio * 0.1) / (client_num - 1))
            for i in range(cfg['classes_size']):
                partition_sizes[i][i % client_num] = non_iid_ratio * 0.1
        else:
            raise ValueError('Non-IID ratio is too large')
    # non-iid
    # each client misses some classes of data, while the other classes of data are distributed uniformly
    elif partition_pattern == 2:
        if 0 < non_iid_ratio < 10:
            # calculate how many classes of data each worker is missing
            missing_class_num = int(round(cfg['classes_size'] * (non_iid_ratio * 0.1)))

            partition_sizes = np.ones((cfg['classes_size'], client_num))
            begin_idx = 0
            for worker_idx in range(client_num):
                for i in range(missing_class_num):
                    partition_sizes[(begin_idx + i) % cfg['classes_size']][worker_idx] = 0.
                begin_idx = (begin_idx + missing_class_num) % cfg['classes_size']

            for i in range(cfg['classes_size']):
                count = np.count_nonzero(partition_sizes[i])
                for j in range(client_num):
                    if partition_sizes[i][j] == 1.:
                        partition_sizes[i][j] = 1. / count
        else:
            raise ValueError('Non-IID ratio is too large')
    else:
        raise ValueError('Not valid partition pattern')

    train_data_partition = datasets.LabelwisePartitioner(
        train_dataset, partition_sizes=partition_sizes, seed=cfg['data_partition_seed']
    )

    return train_data_partition, partition_sizes


if __name__ == "__main__":
    main()
