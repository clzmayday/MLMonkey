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
Feature_List = []

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


# Load Feature list
# Input: Test data
def load_feature_list(data):
    global Feature_List
    Feature_List = data


# Check all data is correctly loaded
def check_data():
    global Feature_Data, Label_Data
    if Feature_Data is None or len(Feature_Data) <= 0:
        logging.error("Feature_Data is missing or not correctly loaded")
    if Label_Data is None or len(Label_Data) <= 0:
        logging.error("Label_Data is missing or not correctly loaded")


def born(data, label="windtex"):
    global Feature_Data, Label_Data, Feature_List
    feature = list(data["1"].keys())
    Feature_List = feature
    f_data = []
    l_data = []
    for i in tqdm(data, desc="Borning Monkey"):
        f_data.append([data[i][r] for r in data[i] if r != label])
        l_data.append(data[i][label])
    Feature_Data = f_data
    Label_Data = l_data
    check_data()


def grow():
    global Feature_Data, Label_Data, Feature_List
    print(Feature_Data)
    print(Feature_List)
    print(Label_Data)


def work():
    pass