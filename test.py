import csv, os
from MLMonkey import FeatureExtraction, Windtex, WindtexModel

# Read CSV files containing the all defect information

data = Windtex.prepare("./Calculator.csv", "./data", "./label_2.json")

WindtexModel.born(data)
# f = WindtexModel.Feature_Data
# fl = WindtexModel.Feature_List
# l = WindtexModel.Label_Data
# csv_data = []
# csv_data.append(fl+["windtex"])
# for i in range(len(f)):
#     a = list(f[i])
#     b = l[i]
#     c = a + [b]
#     csv_data.append(c)

#
# with open("./data.csv", "w") as file:
#     csv_w = csv.writer(file)
#     csv_w.writerows(csv_data)
#     file.close()

from sklearn.tree import DecisionTreeClassifier as model
trained, trained_ex, valid = WindtexModel.grow(model=model())

print()
# WindtexModel.work(trained, trained_ex, [])

