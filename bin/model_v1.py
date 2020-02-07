from sys import argv
import pandas as pd
import numpy as np
import tensorflow as tf
import os


def read_data(x_file_path, y_file_path):
    x = np.load(x_file_path)
    y = np.load(y_file_path)
    #x = np.divide(x, 255, dtype=np.float32)
    return x, y


def read_data_batch(data_dir, y_target):
    y_target_map = {'cd': 'consonant_diacritic', 'gr': 'grapheme_root', 'vd': 'vowel_diacritic'}
    x_data_list = []
    y_data_list = []
    for batch in range(4):
        x_path = os.path.join(data_dir, 'train_x_{}.npy'.format(batch))
        y_path = os.path.join(data_dir, 'train_y_{}-{}.npy'.format(batch, y_target_map[y_target]))
        x_data_list.append(np.load(x_path))
        y_data_list.append(np.load(y_path))
    x = np.concatenate(x_data_list, axis=0)
    y = np.concatenate(y_data_list, axis=0)
    return x, y


def train_valid(data, partition=0.97):
    train_partition = int(partition * data.shape[0])
    train_data = data[:train_partition]
    valid_data = data[train_partition:]
    return train_data, valid_data


def simple_model(x, y):
    x_train, x_valid = train_valid(x)
    y_train, y_valid = train_valid(y)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(8, (7, 7), activation='relu', input_shape=(137, 236, 1)))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_valid, y_valid))



if __name__ == '__main__':
    opt = argv[1]
    args = argv[1:]
    i = 0
    feature_file_path = None
    label_file_path = None
    target = None
    batch_dir = None
    while i < len(args):
        if args[i] == '-x':
            feature_file_path = args[i+1]
        elif args[i] == '-y':
            label_file_path = args[i+1]
        elif args[i] == '-t':
            target = args[i+1]
        elif args[i] == '-dir':
            batch_dir = args[i+1]
        i += 1
    if feature_file_path and label_file_path:
        X, Y = read_data(feature_file_path, label_file_path)
    elif target and batch_dir:
        X, Y = read_data_batch(batch_dir, target)

    simple_model(X, Y)

