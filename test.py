import csv, os
from MLMonkey import FeatureExtraction, Windtex, WindtexModel

# Read CSV files containing the all defect information

data = Windtex.prepare("./Calculator.csv", "./data", "./label_2.json")

WindtexModel.born(data)
from sklearn.naive_bayes import MultinomialNB as model
from sklearn.naive_bayes import MultinomialNB as model_ex
trained, trained_ex, valid = WindtexModel.grow(model=model(), correct_model=None)

WindtexModel.work(trained, trained_ex, [])

