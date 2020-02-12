from sys import argv
import pyarrow.parquet as pq
import pandas as pd
import os
import numpy as np
import cv2


def parquet_to_df(file_path):
    data = pq.read_table(file_path)
    file_name = file_path.split('\\')[-1].split('.')[0]
    data_df = data.to_pandas()
    os.makedirs(os.path.join('..', 'data', 'data_df'), exist_ok=True)
    data_df.to_pickle(os.path.join('..', 'data', 'data_df', file_name))


def plot_image(file_path):
    data = pd.read_pickle(file_path)
    for image_idx in range(len(data)):
        image_data = np.float32(data.iloc[image_idx, 1:].values.reshape(137, 236, 1))
        image_id = data['image_id'][image_idx]
        cv2.imshow(image_id, image_data)
        cv2.waitKey(0)


def transform2model_input(data_label, data_feature, batch_no):
    output_path = os.path.join('..', 'data', 'model_input', 'train')
    os.makedirs(output_path, exist_ok=True)
    data_feature_id = data_feature['image_id']
    #image_array = data_feature[np.setdiff1d(data_feature.columns, 'image_id')].values.reshape(len(data_feature_id),
    #                                                                                          137, 236, 1)
    #np.save(os.path.join(output_path, 'test_x_{}'.format(batch_no)), image_array)
    data_label = data_label[data_label['image_id'].isin(data_feature_id)]
    all_y = None
    for each_y in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:
        y = data_label[each_y]
        one_hot_y = pd.get_dummies(y).values
        if all_y is None:
            all_y = one_hot_y
        else:
            all_y = np.concatenate([all_y, one_hot_y], axis=1)
        #np.save(os.path.join(output_path, 'test_y_{}-{}'.format(batch_no, each_y)), one_hot_y)
    np.save(os.path.join(output_path, 'train_y_{}-all'.format(batch_no)), all_y)

if __name__ == '__main__':
    input_opts = argv[:]
    i = 0
    label_file = None
    feature_file = None
    while i < len(input_opts):
        if input_opts[i] == '-pq':
            pq_file_path = input_opts[i+1]
            parquet_to_df(pq_file_path)
        elif input_opts[i] == '-y':
            label_file = input_opts[i+1]
        elif input_opts[i] == '-x':
            feature_file = input_opts[i+1]
        i += 1
    if label_file and feature_file:
        raw_label = pd.read_csv(label_file, delimiter=',')
        raw_feature = pd.read_pickle(feature_file)
        transform2model_input(raw_label, raw_feature, feature_file.split('_')[-1])