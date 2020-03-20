import numpy as np
import os
import pyarrow.parquet as pq
import tensorflow as tf
import pandas as pd
import cv2


def parquet_to_np(dir_path):
    all_np_list = []
    img_id_list = []
    for i in range(2):
        file_path = os.path.join(dir_path, 'test_image_data_{}.parquet'.format(i))
        data = pq.read_table(file_path)
        data_df = data.to_pandas()
        img_id = data_df['image_id'].values
        print(img_id)
        img_id_list.append(img_id)
        data_df.drop(columns=['image_id'], inplace=True)
        data_np = data_df.values.reshape(len(data_df), 137, 236, 1)
        all_np_list.append(data_np)
    img_id_np = np.concatenate(img_id_list)
    print(img_id_np)
    all_np = np.concatenate(all_np_list)
    return img_id_np, all_np


def predict(model_dir, img_id, img_data):

    models = {}
    batch_size = 4
    batch_num = len(img_id) // batch_size
    for target in ['cd', 'gr', 'vd']:
        model_path = os.path.join(model_dir, '{}_model.h5'.format(target))
        models[target] = tf.keras.models.load_model(model_path)

    results_list = []
    for batch in range(batch_num):
        batch_data = img_data[batch * batch_size:(batch + 1) * batch_size]
        batch_cd_output = models['cd'].predict(batch_data)
        batch_gr_output = models['gr'].predict(batch_data)
        batch_vd_output = models['vd'].predict(batch_data)
        batch_cd_predict = np.argmax(batch_cd_output, axis=1)
        batch_gr_predict = np.argmax(batch_gr_output, axis=1)
        batch_vd_predict = np.argmax(batch_vd_output, axis=1)
        batch_results = np.stack([batch_cd_predict, batch_gr_predict, batch_vd_predict], axis=1)
        results_list.append(batch_results)
    if len(img_id) % batch_size != 0:
        batch_data = img_data[batch * batch_size:]
        batch_cd_output = models['cd'].predict(batch_data)
        batch_gr_output = models['gr'].predict(batch_data)
        batch_vd_output = models['vd'].predict(batch_data)
        batch_cd_predict = np.argmax(batch_cd_output, axis=1)
        batch_gr_predict = np.argmax(batch_gr_output, axis=1)
        batch_vd_predict = np.argmax(batch_vd_output, axis=1)
        batch_results = np.stack([batch_cd_predict, batch_gr_predict, batch_vd_predict], axis=1)
        results_list.append(batch_results)
    all_result_np = np.concatenate(results_list)
    all_result_np_flatten = all_result_np.reshape(-1)
    print(all_result_np_flatten)
    return all_result_np_flatten


def parquet_to_np_v3(dir_path, r_min, r_max):
    all_np = []
    img_id_list = []
    for i in range(r_min, r_max):
        print('batch: {}'.format(i))
        data = pd.read_parquet(os.path.join(dir_path, 'test_image_data_{}.parquet'.format(i)))
        img_id_list.extend(data['image_id'].tolist())
        data = crop_and_resize(data)
        all_np.append(data)
    all_np = np.concatenate(all_np)
    return img_id_list, all_np


def crop_and_resize(image_data, resize=64):
    data = []
    error_id = []
    for i in range(image_data.shape[0]):
        img = image_data.iloc[i, 1:].values.reshape(137, 236, 1)
        ori_img = img.copy()
        _, img = cv2.threshold(img.astype(np.uint8), 30, 255, cv2.THRESH_OTSU)
        canny_img = cv2.Canny(img, 150, 255)
        x, y, w, h = cv2.boundingRect(canny_img)
        img = img[y:y + h, x:x + w]
        try:
            img = cv2.resize(img, (resize, resize)).reshape(64, 64, 1)
        except:
            img = cv2.resize(ori_img, (resize, resize)).reshape(64, 64, 1)
            error_id.append(image_data.iloc[i, 0])
        data.append(img)
    print(error_id)
    data = np.stack(data)
    return data


def predict_v3(model_path, img_id, img_data):
    batch_size = 512
    batch_num = len(img_id) // batch_size
    model = tf.keras.models.load_model(model_path)
    # model.summary()
    results_list = []
    for batch in range(batch_num):
        batch_data = img_data[batch * batch_size:(batch + 1) * batch_size]
        results = model.predict(batch_data)
        results = [np.argmax(n, axis=1) for n in results]
        results = np.stack(results, axis=1)
        results_list.append(results)
    if len(img_id) % batch_size != 0:
        batch_data = img_data[batch_num * batch_size:]
        results = model.predict(batch_data)
        results = [np.argmax(n, axis=1) for n in results]
        results = np.stack(results, axis=1)
        results_list.append(results)
    results_list = np.concatenate(results_list)
    results_list = results_list.reshape(-1)
    return results_list


def transform_to_submission(id_data, target_data):
    id_col = ['{}_{}'.format(img_id, target) for img_id in id_data for target in
              ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']]
    df = pd.DataFrame(zip(id_col, target_data), columns=['row_id', 'target'])
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    test_id, test_data = parquet_to_np(os.path.join('..', 'src'))
    predict_results = predict(os.path.join('..', 'model', 'submit'), test_id, test_data)
    transform_to_submission(test_id, predict_results)