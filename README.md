# Framework of federated learning with client selection using MPI
# 联邦学习客户端选择基础框架

## Introduction

This is an implementation of federated learning framework with client selection using MPI. You can easily extend this basic framework to meet your specific requirements.

* Global hyperparameters are defined in config.yml
* server_main.py is the main file to start the server program
* client_main.py is the main file to start the client program

## Start program

* If you want to run this program that contains up to 10 selected clients, you can input this command in the console:

  ``
  mpiexec --oversubscribe -n 1 python server_main.py : -n 10 python client_main.py
  ``

  Each client and the server run as a process, which communicate with others through MPI.

* Make sure that the maximum number of selected clients is less than or equal to that of clients in the command.

  For example, this command allows up to 100 selected clients in each epoch:
  ``
  mpiexec --oversubscribe -n 1 python server_main.py : -n 100 python client_main.py
  ``

## Results

### Performance on the CIFAR10

* lr= 0.1, decay_rate = 0.993, min_lr = 0.001, local_batch_size = 32, local_iters = 50 for all models.

* The total number of clients in "VGG9" and "VGG9+" is 10 and all clients are selected in each epoch. 
  
  The total number of clients in "VGG9(100-10)" is 100 and 10 clients are randomly selected in each epoch.

* momentum = 0.9, weight_decay = 0.0005 both for "VGG9+" and "VGG9(100-10)".

#### IID (left), Non-IID (data_partition_parttern=2, non_iid_ratio=2, middle) and Non-IID (data_partition_parttern=2, non_iid_ratio=4, right)
<img src="https://github.com/slwang-ustc/FL_PS_MPI_client_selection/blob/main/figs/vgg9_cifar10.png" width="30%">   <img src="https://github.com/slwang-ustc/FL_PS_MPI_client_selection/blob/main/figs/vgg9_cifar10_noniid22.png" width="30.5%">   <img src="https://github.com/slwang-ustc/FL_PS_MPI_client_selection/blob/main/figs/vgg9_cifar10_noniid24.png" width="31%">

## News
An important bug was fixed on June 3, 2023, Please download the latest version.

The third non-iid data partition method is added in the server_main.py on June 16, 2023.
