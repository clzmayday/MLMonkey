import csv, os, pickle
import numpy as np
import sklearn.cluster

from MLMonkey import FeatureExtraction, Windtex, WindtexModel
from matplotlib import pyplot as plt

# Read CSV files containing the all defect information

# data = Windtex.prepare("./Calculator.csv", "./data", "./label_2.json")
# with open("./data.pkl", "wb") as pk_file:
#     pickle.dump(data, pk_file)
#     pk_file.close()
data = None
with open("./data.pkl", "rb") as pk_file:
    data = pickle.load(pk_file)
    pk_file.close()
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

from sklearn.ensemble import AdaBoostRegressor as model
trained, trained_ex, valid = WindtexModel.grow(model=model(learning_rate=0.00001, n_estimators=400))
for i in ["self", "LOO", "RV"]:
    all_result = [[valid[i]["true"][j], valid[i]["predict"][j]] for j in range(len(valid[i]["predict"]))]
    all_result = list(sorted(all_result, key=lambda x: x[0]))

    plt.plot([j for j in range(len(all_result))], [j[1] for j in all_result], "r.")
    plt.plot([j for j in range(len(all_result))], [j[0] for j in all_result], "b.")
    plt.savefig("../Result/"+model.__name__+"_"+i+".jpg")
    plt.close()
    print()
print()
# WindtexModel.work(trained, trained_ex, [])

