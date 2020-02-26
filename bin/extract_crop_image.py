import pandas as pd
from sys import argv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def crop_and_resize(image_data, resize=100):
    data = []
    error_id = []
    for i in tqdm(range(image_data.shape[0]), position=0, leave=True):
        img = image_data.iloc[i, 1:].values.reshape(137, 236, 1)
        ori_img = img.copy()
        _, img = cv2.threshold(img.astype(np.uint8), 30, 255, cv2.THRESH_OTSU)
        canny_img = cv2.Canny(img, 150, 255)
        x, y, w, h = cv2.boundingRect(canny_img)
        img = img[y:y+h, x:x+w]
        try:
            img = cv2.resize(img, (resize, resize))
        except:
            img = cv2.resize(ori_img, (resize, resize))
            error_id.append(image_data.iloc[i, 0])
        data.append(img)
        #cv2.imshow('t', img)
        #cv2.waitKey(0)
    print(error_id)
    data = np.stack(data).reshape(image_data.shape[0], -1)
    df = pd.DataFrame(data)
    df.insert(0, 'image_id', image_data['image_id'].values)
    df.columns = df.columns.astype(str)
    return df

img_df = pd.read_pickle(argv[1])
modified_img = crop_and_resize(img_df, 64)
modified_img.to_feather(argv[1]+'.feather')