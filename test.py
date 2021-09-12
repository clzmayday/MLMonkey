import csv, os
from MLMonkey import FeatureExtraction, Windtex

# Read CSV files containing the all defect information

Windtex.readData("./Calculator.csv", "./data", "./label_2.json")

print()