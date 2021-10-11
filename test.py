import csv, os
from MLMonkey import FeatureExtraction, Windtex, WindtexModel

# Read CSV files containing the all defect information

data = Windtex.prepare("./Calculator.csv", "./data", "./label_2.json")

WindtexModel.born(data)
from sklearn.tree import DecisionTreeRegressor as model
from sklearn.tree import DecisionTreeClassifier as model_ex
trained, valid = WindtexModel.grow(model=model(), correct_model=None)

WindtexModel.work(trained, [])

