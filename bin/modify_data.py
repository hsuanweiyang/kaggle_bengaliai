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
    os.makedirs(os.path.join('..', 'src', 'data_df'), exist_ok=True)
    data_df.to_pickle(os.path.join('..', 'src', 'data_df', file_name))


def plot_image(file_path):
    data = pd.read_pickle(file_path)
    for image_idx in range(len(data)):
        image_data = np.float32(data.iloc[image_idx, 1:].values.reshape(137, 236, 1))
        image_id = data['image_id'][image_idx]
        cv2.imshow(image_id, image_data)
        cv2.waitKey(0)

if __name__ == '__main__':
    input_file_path = argv[1]
    parquet_to_df(input_file_path)
    #plot_image(input_file_path)

