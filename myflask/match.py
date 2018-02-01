from __future__ import print_function
import os
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

#import unicodecsv as csv
import csv

from myflask import config
from myflask.db import models


def find_best_match_image(input_image_path):
    return input_image_path


def get_squares(img):
    """
    Take in a photo, return a top, bottom, and central square of the given size
    (centered)
    """
    top_sq = img[110:130, 115:135, :]
    mid_sq = img[165:185, 115:135, :]
    bottom_sq = img[300:320, 115:135, :]
    return (top_sq, mid_sq, bottom_sq)


def get_hsv_histograms(img):
    """
    Takes in an image in HSV color space and returns an array of len = 692
    which is the frequency of pixels with each Hue[0-180], Saturation[0-256],
    and Value[0-256]
    """
    hue_hist = cv2.calcHist([img], [0], None, [180], [0, 180])
    sat_hist = cv2.calcHist([img], [1], None, [256], [0, 256])
    val_hist = cv2.calcHist([img], [2], None, [256], [0, 256])
    hsv_hist = np.concatenate((hue_hist, sat_hist, val_hist))
    return hsv_hist


def extract_image_features(img_file):
    """
    Takes in a path to a folder and returns processed image histogram frequency
    """

    # Create an empty list to store the processed data
    frames = []

    # 1. read in image
    img = cv2.imread(img_file, 1)

    # 2. resize image if not 250(L) x 350(H)
    if img.size[:,:2] != (250, 350):
        img = cv2.resize(img, (250, 350))

    # 3. convert image to HSV colorspace
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 4. get top, mid, and bottom squares
    square_trio = get_squares(img)

    # 5. get histograms from each square
    top_hist = pd.DataFrame(get_hsv_histograms(square_trio[0]), columns=[os.path.basename(img_file)])
    mid_hist = pd.DataFrame(get_hsv_histograms(square_trio[1]), columns=[os.path.basename(img_file)])
    bottom_hist = pd.DataFrame(get_hsv_histograms(square_trio[2]), columns=[os.path.basename(img_file)])
    hist = top_hist.append(mid_hist, ignore_index = True).append(bottom_hist, ignore_index=True).transpose()
    frames.append(hist)

    data_df = pd.concat(frames)
    data_df.to_pickle('test.p')

    return data_df


def load_model(model_path=config.MATCHING_MODEL_PATH):
    return pickle.load(open(model_path, 'rb'))


def find_similar_items(model, input_image_features):
    '''
    Take in an array of probabilities from sklearn model output and return a
    dataframe with the image ID, probability of dress, and array index.
    '''
    probs = model.predict_proba(input_image_features)
    # print(probs)
    probability_list = []

    # HACK: the models were generated with a slightly different ID set,
    # this maps the difference.
    def item_id_for_model_dress_id(model_dress_id):
        return MODEL_ITEM_MAPPING[model_dress_id]

    for dress_id, probability in enumerate(probs[0]):
        # print(dress_id, probability)
        if probability > 0:
            probability_list.append({
                'dress_id': item_id_for_model_dress_id(dress_id),
                'probability': probability,
                })
    return probability_list


def load_model_item_mapping(item_csv_path=config.MATCHING_MODEL_DRESS_DETAILS_PATH):
    '''
    This is a hack to map from the pre-sqlalchemy IDs to the primary key IDs in
    the database. At startup, we load a mapping between the old and new IDs and
    use it when querying for Items/ItemImages.
    '''
    with open(item_csv_path, 'r') as f:
        records = list(csv.reader(f))
        header, records = records[0], records[1:]
        original_recs = [dict(zip(header, rec)) for rec in records]
    
    original_recs_by_url = {r['detail_url']: r for r in original_recs}

    item_mapping = {}
    session = models.Session()
    for item in session.query(models.Item).all():
        mapped_item = original_recs_by_url.get(item.detail_url)
        if not mapped_item:
            print("NO MAPPING FOUND: {}".format(item.detail_url))
            continue
        item_mapping[int(mapped_item['dress_id'])] = item.id
    
    return item_mapping

MODEL_ITEM_MAPPING = load_model_item_mapping()
