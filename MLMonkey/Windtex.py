import csv, os
from tqdm import tqdm
from MLMonkey import FeatureExtraction

windtex_head = []
windtex_data = []


def read_csv_data(filepath):
    global windtex_head, windtex_data
    with open(filepath, "r") as csvFile:
        csvReader = [i for i in csv.reader(csvFile)]
        windtex_head = csvReader[0]
        for i in csvReader[1:]:
            csv_row = {}
            for name in range(len(windtex_head)):
                if windtex_head[name] == "description":
                    csv_row[windtex_head[name]] = i[name].split(",")
                elif windtex_head[name] == "Length (meters)":
                    csv_row[windtex_head[name]] = i[name].split("+")
                else:
                    csv_row[windtex_head[name]] = i[name]
            windtex_data.append(csv_row)
        csvFile.close()

    return windtex_head, windtex_data


def merging_defect():
    pass


def average(values, weights=None):
    avg = 0
    if weights is not None:
        for i in range(len(values)):
            avg += values * (weights[i] / sum(weights))
    else:
        avg = sum(values) / len(values)

    return avg


def readData(windtex_path, image_path, label_path):
    global windtex_head, windtex_data
    w_head, w_data = read_csv_data(windtex_path)

    FeatureExtraction.readImages(image_path)
    FeatureExtraction.readVIALabel(label_path)
    FeatureExtraction.loadData(crop_pad=0.25)
    f_data = FeatureExtraction.featureExtract(distance_threshold=0)
    all_feature = list(FeatureExtraction.get_FeatureRange().keys())
    m_data = dict()
    for damage in tqdm(w_data, desc="Merging Features to Damage"):
        m_data[damage["ID"]] = dict()
        all_defect = list(filter(lambda x: damage["ID"] == x["filename"].split("_")[0], f_data.values()))

    print()