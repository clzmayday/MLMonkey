import logging, os

import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import confusion_matrix as CM
import sklearn.tree as T
import math
from tqdm import tqdm
import copy

Feature_Data = []
Label_Data = []

logging.getLogger().setLevel(logging.INFO)


# Transfer the feature values to defect_id list, feature name list and data list
# Input: label - feature values
# Output: Defect_id:List()
#         Feature name:List()
#         Data:List()
def convert2List(label, feature):
    data = []
    id_list = []
    for i in label.keys():
        id_list.append(i)
        e = []
        for j in feature:
            e.append(label[i][j])
        data.append(e)

    return id_list, feature, data


# Load feature data into model
# Input: Feature data
def load_feature_data(data):
    global Feature_Data
    Feature_Data = data


# Load Test data into model
# Input: Test data
def load_label_data(data):
    global Label_Data
    Label_Data = data


# Check all data is correctly loaded
def check_data():
    global Feature_Data, Label_Data
    if Feature_Data is None or len(Feature_Data) <= 0:
        logging.error("Feature_Data is missing or not correctly loaded")
    if Label_Data is None or len(Label_Data) <= 0:
        logging.error("Label_Data is missing or not correctly loaded")


def born(data):
    feature = data["1"].keys()

    print()