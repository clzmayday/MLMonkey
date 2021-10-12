import logging, os

import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import confusion_matrix as CM
import sklearn.tree as T
from sklearn.model_selection import LeaveOneOut
from sklearn import base
import math
from tqdm import tqdm
import copy

Feature_Data = []
Label_Data = []
Feature_List = []
Food = None
Trained_Model = None

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


def norm_label(data, gap=0.5, align="upper"):
    n_data = []
    if align == "centre":
        for i in data:
            n_data.append(round(i / gap))
    elif align == "upper":
        for i in data:
            n_data.append(int(1 + i / gap))

    elif align == "lower":
        for i in data:
            n_data.append(int(i / gap))

    return np.array(n_data).astype(int)


def born(data, label="windtex", label_norm=True):
    global Feature_Data, Label_Data, Feature_List
    feature = list(data["1"].keys())
    Feature_List = [i for i in feature if i != label]
    f_data = []
    l_data = []
    for i in tqdm(data, desc="Borning Monkey"):
        f_data.append([data[i][r] for r in data[i] if r != label])
        l_data.append(data[i][label])
    f_data = np.array(f_data).astype(int)
    l_data = np.array(l_data).astype(float)
    Feature_Data = f_data
    if label_norm:
        Label_Data = norm_label(l_data)
    else:
        Label_Data = l_data
    check_data()

    return Feature_Data, Label_Data, Feature_List


def evaluate(true, predict):
    result = {"MAE": 0, "RMSE": 0, "MSE": 0, "ACC-0": 0, "ACC-1": 0, "ACC-2": 0, "ACC-3": 0,
              "true":np.array(true), "predict":np.array(predict)}
    for i in range(len(true)):
        gap = abs(true[i] - predict[i])

        if gap <= 0:
            result["ACC-0"] += 1 / len(true)
        if gap <= 1:
            result["ACC-1"] += 1 / len(true)
        if gap <= 2:
            result["ACC-2"] += 1 / len(true)
        if gap <= 3:
            result["ACC-3"] += 1 / len(true)

        result["MAE"] += gap / len(true)
        result["MSE"] += gap ** 2 / len(true)
    result["RMSE"] = math.sqrt(result["MSE"])

    return result


def grow(model=None, correct_model=None, self_validate=True, LOO=True, recursive_validation=30, round_prediction=True):
    global Feature_Data, Label_Data, Feature_List, Trained_Model
    trained_model = None
    trained_model_ex = None
    valid_result = {"self": {}, "LOO": {}, "RV": {}}
    for _ in tqdm(range(1), desc="Growing Monkey"):
        food = None
        food_ex = None
        if model is not None:
            food = base.clone(model)
        if correct_model is not None:
            food_ex = base.clone(correct_model)

        trained_model = food.fit(Feature_Data, Label_Data)
        trained_model_ex = None
        gap = []
        if correct_model is not None:
            true = []
            predict = []
            loo = LeaveOneOut()
            for train, test in loo.split(Feature_Data):
                valid_model = base.clone(model)
                v_trained = valid_model.fit(Feature_Data[train], Label_Data[train])
                true.append(Label_Data[test][0])
                predict.append(v_trained.predict(Feature_Data[test])[0])
            for i in range(len(true)):
                gap.append(true[i] - predict[i])

        else:
            gap = [0 for i in range(len(Label_Data))]
        gap = np.array(gap).astype(int)
        if correct_model is not None:
            trained_model_ex = food_ex.fit(Feature_Data, gap)

        if self_validate:
            predict = trained_model.predict(Feature_Data)
            if round_prediction:
                predict = np.round(predict)
            valid_result["self"] = evaluate(Label_Data, predict)
        if LOO:
            true = []
            predict = []
            loo = LeaveOneOut()
            for train, test in loo.split(Feature_Data):
                valid_model = base.clone(model)
                v_trained = valid_model.fit(Feature_Data[train], Label_Data[train])
                true.append(Label_Data[test][0])
                v_predicted = v_trained.predict(Feature_Data[test])[0]
                predict.append(v_predicted)
            if round_prediction:
                predict = np.round(predict)
            valid_result["LOO"] = evaluate(true, predict)
        if recursive_validation > 0:
            true = []
            predict = []
            for i in range(recursive_validation):

                index = [i for i in range(len(Feature_Data))]
                np.random.shuffle(index)
                train = np.array(index[:int(len(index) * 0.9)])
                test = np.array(index[int(len(index) * 0.9):])
                valid_model = base.clone(model)
                v_trained = valid_model.fit(Feature_Data[train], Label_Data[train])
                true.extend(Label_Data[test])
                v_predicted = v_trained.predict(Feature_Data[test])
                predict.extend(v_predicted)
            if round_prediction:
                predict = np.round(predict)
            valid_result["RV"] = evaluate(true, predict)

    return trained_model, trained_model_ex, valid_result


def work(model, model_ex, feature):
    return model.predict([feature])[0] + model_ex.predict([feature])[0]
