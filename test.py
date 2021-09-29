import csv, os
from MLMonkey import FeatureExtraction, Windtex, WindtexModel

# Read CSV files containing the all defect information

data = Windtex.prepare("./Calculator.csv", "./data", "./label_2.json")

WindtexModel.born(data)
WindtexModel.grow()

print()