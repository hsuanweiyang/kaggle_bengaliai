from sys import argv
import pandas as pd
import numpy as np
import tensorflow as tf


def read_data(x_file_path, y_file_path):
    x = np.load(x_file_path)
    y = np.load(y_file_path)
    x, y = x/255, y/255
    return x, y


def train_valid(data, partition=0.97):
    train_partition = int(partition * data.shape[0])
    train_data = data[:train_partition]
    valid_data = data[train_partition:]
    return train_data, valid_data


if __name__ == '__main__':
    opt = argv[1]
    args = argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '-x':
            feature_file_path = args[i+1]
        elif args[i] == '-y':
            label_file_path = args[i+1]
        i += 1
    X, Y = read_data(feature_file_path, label_file_path)