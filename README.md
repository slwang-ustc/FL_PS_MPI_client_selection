# Framework of federated learning with client selection using MPI

## Introduction

This is an implementation of federated learning framework with client selection using MPI.

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
The data on the clients follows IID.

We set the lr as 0.1, local_batch_size as 32, local_iters as 50, decay_rate as 0.993 and min_lr as 0.001 for all models.

The total number of clients in "VGG9" and "VGG9+" is 10 and all clients are selected to train the model in each epoch. The total number of clients in "VGG9(100-10)" is 100 and 10 clients are randomly selected to train the model in each epoch.

The momentum and weight_decay are 0.9 and 0.0005 both for VGG9+ and VGG9(100-10), respectively.

<img src="https://github.com/slwang-ustc/FL_PS_MPI_client_selection/blob/main/figs/vgg9_cifar10.png" width="50%">

## News
An important bug was fixed on June 3, 2023, Please download the latest version.
