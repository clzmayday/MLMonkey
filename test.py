import csv, os
from MLMonkey import FeatureExtraction, Windtex

# Read CSV files containing the all defect information

data = Windtex.readData("./Calculator.csv", "./data", "./label_2.json")

# csv_data = [list(data["1"].keys())]
# for i in data:
#     row = []
#     for j in data[i]:
#         if j == "description" or j == "position":
#             row.append(",".join(data[i][j]))
#         else:
#             row.append(data[i][j])
#     csv_data.append(row)
#
# with open("./first.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(csv_data)

print()