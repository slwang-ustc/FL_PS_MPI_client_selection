import numpy as np

import datasets
from config import cfg


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


# load the test dataset and test loader
train_dataset, test_dataset = datasets.load_datasets(cfg['dataset_type'], cfg['dataset_path'])
# test_loader = datasets.create_dataloaders(test_dataset, batch_size=cfg['test_batch_size'], shuffle=False)


# partition the dataset
train_data_partition, partition_sizes = partition_data(
    dataset_type=cfg['dataset_type'],
    partition_pattern=cfg['data_partition_pattern'],
    non_iid_ratio=cfg['non_iid_ratio'],
    client_num=10
)

train_loader = datasets.create_dataloaders(
    train_dataset, batch_size=cfg['local_batch_size'], selected_idxs=train_data_partition.use(0)
)

print(train_loader.loader[0])