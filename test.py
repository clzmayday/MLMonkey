import csv, os
from MLMonkey import FeatureExtraction

# Read CSV files containing the all defect information
csv_head = []
csv_data = []
with open ("./Calculator.csv", "r") as csvFile:
    csvReader = [i for i in csv.reader(csvFile)]
    csv_head = csvReader[0]
    for i in csvReader[1:]:
        csv_row = {}
        for name in range(len(csv_head)):
            if csv_head[name] == "description":
                csv_row[csv_head[name]] = i[name].split(",")
            elif csv_head[name] == "Length (meters)":
                csv_row[csv_head[name]] = i[name].split("+")
            else:
                csv_row[csv_head[name]] = i[name]
        csv_data.append(csv_row)
    csvFile.close()


FeatureExtraction.readImages("./data/")
FeatureExtraction.readVIALabel("./label_2.json")

data1 = FeatureExtraction.loadData(crop_pad=0.25)
data2 = FeatureExtraction.featureExtract()
print()