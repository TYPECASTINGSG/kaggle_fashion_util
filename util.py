import pandas as pd
from itertools import groupby
import numpy as np
import cv2

CATEGORY_LEN = 47
WIDTH = 512
HEIGHT = 512

def read_shoes():
    shoes = pd.read_csv('shoes.csv')
    shoes['ClassId'] = shoes['ClassId'].astype(str)
    return shoes

def encode(input_string):
    return [(len(list(g)), k) for k,g in groupby(input_string)]

def run_length(label_vec):
    encode_list = encode(label_vec)
    index = 1
    class_dict = {}
    for i in encode_list:
        if i[1] != CATEGORY_LEN-1:
            if i[1] not in class_dict.keys():
                class_dict[i[1]] = []
            class_dict[i[1]] = class_dict[i[1]] + [index, i[0]]
        index += i[0]

    return class_dict

# dataframe of a single image, multiple classes
# 0-45 classes, 46 background
def make_mask_img(segment_df):
    seg_width = segment_df.at[0, "Width"]
    seg_height = segment_df.at[0, "Height"]

    seg_img = np.full(seg_width*seg_height, CATEGORY_LEN-1, dtype=np.int32)

    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):

        pixel_list = list(map(int, encoded_pixels.split(" ")))
        
        for i in range(0, len(pixel_list), 2):
            start_index = pixel_list[i] - 1
            index_len = pixel_list[i+1] - 1
            seg_img[start_index:start_index+index_len] = int(class_id.split("_")[0])
    
    seg_img = seg_img.reshape((seg_height, seg_width), order='F')
    seg_img = cv2.resize(seg_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    return seg_img