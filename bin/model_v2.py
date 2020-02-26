from sys import argv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, AveragePooling2D, Flatten, ReLU, GlobalAveragePooling2D
from tensorflow.python.platform import gfile
import os


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filter_nums, strides=1, residual_path=False):
        super(ResBlock, self).__init__()
        self.conv_1 = Conv2D(filter_nums, (3, 3), strides=strides, padding='same')
        self.bn_1 = BatchNormalization()
        self.activation = ReLU()
        self.conv_2 = Conv2D(filter_nums, (3, 3), strides=1, padding='same')
        self.bn_2 = BatchNormalization()

        if strides != 1:
            self.block = tf.keras.Sequential()
            self.block.add(Conv2D(filter_nums, (1, 1), strides=strides))
        else:
            self.block = lambda x: x

    def call(self, inputs, training=None):
        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.bn_2(x, training=training)

        identity = self.block(inputs)
        outputs = tf.keras.layers.add([x, identity])
        outputs = tf.nn.relu(outputs)
        return outputs


class ResNet(tf.keras.Model):
    def __init__(self, layers_dims, num_class):
        super(ResNet, self).__init__()
        self.model = tf.keras.Sequential(
            [
                Conv2D(64, (3, 3), strides=(1, 1)),
                BatchNormalization(),
                ReLU(),
                MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')
            ]
        )

        self.layer_1 = self.ResNet_build(64, layers_dims[0])
        self.layer_2 = self.ResNet_build(128, layers_dims[1], strides=2)
        self.layer_3 = self.ResNet_build(256, layers_dims[2], strides=2)
        self.layer_4 = self.ResNet_build(512, layers_dims[3], strides=2)
        self.avg_pool = GlobalAveragePooling2D()
        self.fc_model = Dense(num_class, activation='softmax')

    def call(self, inputs, training=None):
        x = self.model(inputs)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avg_pool(x)
        x = self.fc_model(x)
        return x

    def ResNet_build(self, filter_nums, block_nums, strides=1):
        build_model = tf.keras.Sequential()
        build_model.add(ResBlock(filter_nums, strides))
        for _ in range(1, block_nums):
            build_model.add(ResBlock(filter_nums, strides=1))
        return build_model


def ResNet18(num_class):
    return ResNet([2, 2, 2, 2], num_class)


def read_data_batch(data_dir, y_target):
    y_target_map = {'cd': 'consonant_diacritic', 'gr': 'grapheme_root', 'vd': 'vowel_diacritic', 'a': 'all'}
    x_data_list = []
    y_data_list = []
    for batch in range(1):
        x_path = os.path.join(data_dir, 'train_x_{}.npy'.format(batch))
        y_path = os.path.join(data_dir, 'train_y_{}-{}.npy'.format(batch, y_target_map[y_target]))
        x_data_list.append(np.load(x_path).astype(np.float32))
        y_data_list.append(np.load(y_path).astype(np.float32))
    x = np.concatenate(x_data_list, axis=0)
    y = np.concatenate(y_data_list, axis=0)
    return x, y


def train_valid(data, partition=0.97):
    train_partition = int(partition * data.shape[0])
    train_data = data[:train_partition]
    valid_data = data[train_partition:]
    return train_data, valid_data


def train(x, y, y_target, saved_checkpoint=None):
    y_target_map = {'cd': 7, 'gr': 168, 'vd': 11, 'a': 186}
    x_train, x_valid = train_valid(x)
    y_train, y_valid = train_valid(y)
    model = ResNet18(y_target_map[y_target])
    model.build(input_shape=(None, 137, 236, 1))
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=1, batch_size=1)


def predict(model_path):
    test = np.random.rand(1, 137, 236, 1)
    model = tf.keras.models.load_model(model_path)
    a = model.predict(test)
    print(a)
    #model.summary()


if __name__ == '__main__':
    target = argv[1]
    predict(target)
    exit()
    batch_dir = argv[2]
    X, Y = read_data_batch(batch_dir, target)
    train(X, Y, target)