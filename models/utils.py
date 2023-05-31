import torch
import torch.nn as nn
from models.resnet import resnet18
from models.cnn import CNN
from config import cfg


def create_model_instance(model_type):
    data_shape = cfg['data_shape']
    hidden_size = cfg['cnn_hidden_size']
    classes_size = cfg['classes_size']
    if model_type == 'cnn':
        return CNN(data_shape, hidden_size, classes_size)
    if model_type == 'resnet18':
        return resnet18()
    else:
        raise ValueError("Not valid model type")
