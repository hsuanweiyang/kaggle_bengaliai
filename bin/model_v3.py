from sys import argv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, AveragePooling2D, Flatten, ReLU, GlobalAveragePooling2D, Dropout
import os
from collections import defaultdict
from datetime import datetime


def res_block(inputs, filter_nums, strides=1):
    conv_1 = Conv2D(filter_nums, (3, 3), strides=strides, padding='same',
                    kernel_regularizer=tf.keras.regularizers.l2(l=0.02))
    bn_1 = BatchNormalization()
    activation = ReLU()
    # dropout = Dropout(rate=0.4)
    conv_2 = Conv2D(filter_nums, (3, 3), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=0.02))
    bn_2 = BatchNormalization()
    output = conv_1(inputs)
    output = bn_1(output, training=True)
    output = activation(output)
    # output = dropout(output, training=True)
    output = conv_2(output)
    output = bn_2(output, training=True)
    if strides != 1:
        short_path = Conv2D(filter_nums, (1, 1), strides=strides)
        identity = short_path(inputs)
    else:
        identity = inputs

    output = tf.keras.layers.add([output, identity])
    output = tf.nn.relu(output)
    return output


def ResNet(layers_dim, model_name='resnet'):
    inputs = tf.keras.Input(shape=(64, 64, 1))
    x = Conv2D(64, (3, 3), strides=(1, 1))(inputs)
    x = BatchNormalization()(x, training=True)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
    for _ in range(layers_dim[0]):
      x = res_block(x, 64)
      x = res_block(x, 64)
    for _ in range(layers_dim[1]):
      x = res_block(x, 128, 2)
      x = res_block(x, 128)
    for _ in range(layers_dim[2]):
      x = res_block(x, 256, 2)
      x = res_block(x, 256)
    for _ in range(layers_dim[3]):
      x = res_block(x, 512, 2)
      x = res_block(x, 512)
    x = GlobalAveragePooling2D()(x)
    cd_output = Dense(7, activation='softmax')(x)
    gr_output = Dense(168, activation='softmax')(x)
    vd_output = Dense(11, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=[cd_output, gr_output, vd_output], name=model_name)
    tf.keras.utils.plot_model(model, to_file=os.path.join('model_{}'.format(model_name), '{}-model.png'.format(model_name)))
    return model


def read_data_batch(data_dir, y_target):
    y_target_map = {'cd': 'consonant_diacritic', 'gr': 'grapheme_root', 'vd': 'vowel_diacritic'}
    x_data_list = []
    y_data_dict = defaultdict(list)
    for batch in range(4):
        x_path = os.path.join(data_dir, 'feather', 'train_image_data_{}.feather'.format(batch))
        x_data = pd.read_feather(x_path)
        x_data_list.append(x_data.iloc[:, 1:].values.reshape(-1, 64, 64, 1))
        if y_target == 'a':
            for each_ta in ['cd', 'gr', 'vd']:
                y_path = os.path.join(data_dir, 'np_array', 'train_y_{}-{}.npy'.format(batch, y_target_map[each_ta]))
                y_data_dict[each_ta].append(np.load(y_path).astype(np.float32))
        else:
            y_path = os.path.join(data_dir, 'np_array', 'train_y_{}-{}.npy'.format(batch, y_target_map[y_target]))
            y_data_dict[y_target].append(np.load(y_path).astype(np.float32))
    x = np.concatenate(x_data_list, axis=0).astype(np.float32)
    if y_target == 'a':
        y = [np.concatenate(y_data_dict[ta], axis=0) for ta in ['cd', 'gr', 'vd']]
    else:
        y = np.concatenate(y_data_dict[y_target], axis=0)
    return x, y


def train_valid_split(data, partition=0.97):
    train_partition = int(partition * data.shape[0])
    train_data = data[:train_partition]
    valid_data = data[train_partition:]
    return train_data, valid_data


def train(x, y, model_path=None):
  model_name = 'resnet-18'
  tf.keras.backend.set_learning_phase(True)
  os.makedirs(os.path.join('model_{}'.format(model_name)), exist_ok=True)
  train_x, valid_x = train_valid_split(x)
  train_y_0, valid_y_0 = train_valid_split(y[0])
  train_y_1, valid_y_1 = train_valid_split(y[1])
  train_y_2, valid_y_2 = train_valid_split(y[2])
  if model_path is None:
    model = ResNet([2, 2, 2, 2], model_name)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
  else:
    model = tf.keras.models.load_model(model_path)
  model.summary()
  model.fit(train_x, [train_y_0, train_y_1, train_y_2], epochs=10, batch_size=64, validation_data=(valid_x, [valid_y_0, valid_y_1, valid_y_2]))
  model.save(os.path.join('model_{}'.format(model_name), 'model_{}.h5'.format(datetime.now().strftime('%Y%m%d-%H%M%S')))


def predict(test, model_path):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    results = model.predict(test)
    return results


if __name__ == '__main__':
    target = argv[1]
    batch_dir = argv[2]
    X, Y = read_data_batch(batch_dir, target)
    train(X, Y)