from sys import argv
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from datetime import datetime


def read_data(x_file_path, y_file_path):
    x = np.load(x_file_path)
    y = np.load(y_file_path)
    #x = np.divide(x, 255, dtype=np.float32)
    return x, y


def read_data_batch(data_dir, y_target):
    y_target_map = {'cd': 'consonant_diacritic', 'gr': 'grapheme_root', 'vd': 'vowel_diacritic', 'a': 'all'}
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


def custom_loss(y_true, y_pred):
    base = 0
    softmax_pred_list = []
    for flag in [168, 11, 7]:
        softmax_pred_list.append(tf.nn.softmax(y_pred[:, base:base + flag]))
        base = base + flag
    y_pred = tf.concat(softmax_pred_list, axis=1)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)


def train(x, y, y_target, saved_checkpoint=None):
    x_train, x_valid = train_valid(x)
    y_train, y_valid = train_valid(y)
    model = create_model(y_target)
    model.summary()
    '''
    if saved_checkpoint is not None:
        model.load_weights(saved_checkpoint)
    ck_path = os.path.join('..', 'model_ck', 'ck_{}-{}'.format(y_target, datetime.now().strftime('%Y%m%d-%H%M%S')))
    os.makedirs(ck_path)
    callback_cp = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ck_path, '{epoch:04d}'), verbose=1, period=1, save_weights_only=True)
    '''
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_valid, y_valid), )
    output_dir = os.path.join('..', 'model', '{}-{}'.format(y_target, datetime.now().strftime('%Y%m%d-%H%M%S')))
    os.makedirs(output_dir)
    model.save(os.path.join(output_dir, '{}_model.h5'.format(y_target)))


def re_train(x, y, y_target, model_path):
    x_train, x_valid = train_valid(x)
    y_train, y_valid = train_valid(y)
    model = tf.keras.models.load_model(model_path)
    print(model.summary())
    model.fit(x_train, y_train, epochs=6, batch_size=32, validation_data=(x_valid, y_valid))
    output_dir = os.path.join('..', 'model', 're_{}-{}'.format(y_target, datetime.now().strftime('%Y%m%d-%H%M%S')))
    os.makedirs(output_dir)
    model.save(os.path.join(output_dir, '{}_model.h5'.format(y_target)))


def create_model(y_target):
    y_target_map = {'cd': 7, 'gr': 168, 'vd': 11, 'a': 186}
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (7, 7), activation='relu', input_shape=(137, 236, 1)))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(y_target_map[y_target], activation='softmax'))
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


def predict(feature_file, model_file):
    feature_data = np.load(feature_file)
    model = tf.keras.models.load_model(model_file)
    results = model.predict(feature_data)
    return results


if __name__ == '__main__':
    opt = argv[1]
    args = argv[1:]
    i = 0
    feature_file_path = None
    label_file_path = None
    target = None
    batch_dir = None
    model_checkpoint = None
    cd_model = None
    gr_model = None
    vd_model = None
    test_file = None
    retrain = False
    while i < len(args):
        if args[i] == '-x':
            feature_file_path = args[i+1]
        elif args[i] == '-y':
            label_file_path = args[i+1]
        elif args[i] == '-ta':
            target = args[i+1]
        elif args[i] == '-dir':
            batch_dir = args[i+1]
        elif args[i] == '-ck':
            model_checkpoint = args[i+1]
        elif args[i] == '-mcd':
            cd_model = args[i+1]
        elif args[i] == '-mgr':
            gr_model = args[i+1]
        elif args[i] == '-mvd':
            vd_model = args[i+1]
        elif args[i] == '-te':
            test_file = args[i+1]
        elif args[i] == '-re':
            retrain = True

        i += 1
    if opt == '--train':
        if feature_file_path and label_file_path:
            X, Y = read_data(feature_file_path, label_file_path)
        elif target and batch_dir:
            X, Y = read_data_batch(batch_dir, target)
        if retrain:
            if cd_model:
                model_path = cd_model
            elif gr_model:
                model_path = gr_model
            elif vd_model:
                model_path = vd_model
            re_train(X, Y, target, model_path)
        else:
            train(X, Y, target)
    elif opt == '--predict':
        if cd_model and gr_model and vd_model:
            cd_result = predict(test_file, cd_model)
            gr_result = predict(test_file, gr_model)
            vd_result = predict(test_file, vd_model)
            cd_result = np.argmax(cd_result, axis=1)
            gr_result = np.argmax(gr_result, axis=1)
            vd_result = np.argmax(vd_result, axis=1)
            all_result = np.stack([cd_result, gr_result, vd_result], axis=1)
            flatten_result = all_result.reshape(-1)
            sample_data = pd.read_csv(os.path.join('..', 'src', 'sample_submission.csv'))
            sample_data['target'] = flatten_result
            os.makedirs(os.path.join('..', 'submission'), exist_ok=True)
            sample_data.to_csv(os.path.join('..', 'submission', 'submit_{}.csv'.format(datetime.now().strftime('%Y%m%d-%H%M%S'))), index=False)





