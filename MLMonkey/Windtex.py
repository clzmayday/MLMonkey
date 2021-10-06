import csv, os, os
from tqdm import tqdm
from MLMonkey import FeatureExtraction
from PIL import Image

windtex_head = []
windtex_data = []
feature_range = None


def read_csv_data(filepath):
    global windtex_head, windtex_data
    with open(filepath, "r") as csvFile:
        csvReader = [i for i in csv.reader(csvFile)]
        windtex_head = csvReader[0]
        for i in csvReader[1:]:
            csv_row = {}
            for name in range(len(windtex_head)):
                if windtex_head[name] == "description" or windtex_head[name] == "Position":
                    csv_row[windtex_head[name]] = i[name].split(",")
                elif windtex_head[name] == "Length (meters)":
                    csv_row[windtex_head[name]] = i[name].split("+")
                else:
                    csv_row[windtex_head[name]] = i[name]
            windtex_data.append(csv_row)
        csvFile.close()

    return windtex_head, windtex_data


def merging_defect(feature_list, feature, size, range_extend=False):

    data = dict()
    weight = [i["a"] for i in size]
    for fea in feature_list:
        if fea == "distance":
            continue
        f_list = [i[fea] for i in feature]
        avg = 0
        if fea == "size":
            data[fea] = round(sum(weight), 1)
            continue
        else:
            avg = average(f_list, weight)
        if range_extend:
            data[fea] = round(avg*2)
        else:
            data[fea] = round(avg)

    return data


def average(values, weights=None):
    avg = 0
    if weights is not None:
        for i in range(len(values)):
            avg += values[i] * (weights[i] / sum(weights))
    else:
        avg = sum(values) / len(values)

    return avg


def quantify_desc(desc):
    desc_dict = {"erosion":0,
                 "coat_protection":0,
                 "laminate_vis_dry_dam":0,
                 "hole/crack/bonding":0,
                 "lightning_recep_dam/impact":0,
                 "assist":0,
                 "other":0}
    for i in desc:
        if "foil" in i or "protection" in i:
            desc_dict["coat_protection"] = 2
        elif "erosion" in i:
            if "class" in i:
                desc_dict["erosion"] = int(i[-1])
            else:
                desc_dict["erosion"] = 1
        elif "coat" in i:
            if desc_dict["coat_protection"] != 0:
                continue
            desc_dict["coat_protection"] = 1
        elif "laminate" in i:
            if "visible" in i:
                desc_dict["laminate_vis_dry_dam"] = 1
            elif "dry" in i:
                desc_dict["laminate_vis_dry_dam"] = 2
            elif "damage" in i:
                desc_dict["laminate_vis_dry_dam"] = 3
        elif "drain" in i or "rain" in i or "vortex" in i:
            desc_dict["assist"] = 1
        elif "lightning" in i or "impact" in i:
            if "receptor" in i:
                desc_dict["lightning_recep_dam/impact"] = 1
            elif "lightning" in i:
                desc_dict["lightning_recep_dam/impact"] = 2
            elif "impact" in i:
                desc_dict["lightning_recep_dam/impact"] = 3
        elif "hole" in i or "crack" in i or "bond" in i:
            if "hole" in i:
                desc_dict["hole/crack/bonding"] = 1
            elif "crack" in i:
                desc_dict["hole/crack/bonding"] = 2
            elif "bond" in i:
                desc_dict["hole/crack/bonding"] = 3
        else:
            desc_dict["other"] = 1

    return desc_dict

def group_norm(data, name, group):
    full = [data[i][name] for i in data]
    up = max(full)
    low = min(full)
    gap = round((up-low) / group)
    norm = {}
    for i in data:
        norm[i] = int((data[i][name] - low) / gap) + 1

    return norm


def prepare(windtex_path, image_path, label_path, range_extend=True):
    global windtex_head, windtex_data, feature_range
    w_head, w_data = read_csv_data(windtex_path)

    FeatureExtraction.readImages(image_path)
    FeatureExtraction.readVIALabel(label_path)
    FeatureExtraction.loadData(crop_pad=0.25)
    f_data = FeatureExtraction.featureExtract(distance_threshold=0)
    all_feature = FeatureExtraction.get_FeatureList()
    feature_range = FeatureExtraction.get_FeatureRange()
    m_data = dict()
    for damage in tqdm(w_data, desc="Merging Features to Damage"):
        m_data[damage["ID"]] = dict()
        all_defect = list(filter(lambda x: damage["ID"] == x["filename"].split("_")[0], f_data.values()))
        file = list(set([i["filename"] for i in all_defect]))
        act_size = dict()
        file = sorted(file)
        for i in range(len(file)):
            im_s = Image.open(os.path.abspath(image_path) + "/" + file[i]).height
            act_size[file[i]] = float(damage["Length (meters)"][i]) / im_s

        # fixed data
        m_data[damage["ID"]]['damage_qty'] = int(damage['Damage Qty Per Meter'])
        # m_data[damage["ID"]]['position'] = damage['Position']
        m_data[damage["ID"]]['location'] = float(damage['Damage Location'])
        d_dict = quantify_desc(damage['description'])
        for d in d_dict:
            m_data[damage["ID"]][d] = d_dict[d]
        m_data[damage["ID"]]['num_desc'] = len(damage['description'])
        m_data[damage["ID"]]['continuous'] = 0
        # loop
        size_list = []
        feature_list = []

        for i in all_defect:
            if i["boundary"][0][0] <= 10 or i["boundary"][0][1] <= 10 or i["boundary"][1][0] - i["boundary"][0][2] \
                    <= 10 or i["boundary"][1][1] - i["boundary"][0][3] <= 10:
                m_data[damage["ID"]]['continuous'] = 1
            size_dict = {"w": round(act_size[i["filename"]] * i["width"] * 100, 4),
                         "h": round(act_size[i["filename"]] * i["height"] * 100, 4),
                         "a": round(act_size[i["filename"]] * act_size[i["filename"]] * i["poly_size"] * 10000, 4)}
            feature_dict = dict()
            for f in all_feature:
                feature_dict[f] = i[f]
            size_list.append(size_dict)
            feature_list.append(feature_dict)

        merged_data = merging_defect(all_feature, feature_list, size_list, range_extend=range_extend)
        for i in merged_data:
            m_data[damage["ID"]][i] = merged_data[i]
        # m_data[damage["ID"]]["size"] = round(sum([i["a"] for i in size_list]), 1)
        m_data[damage["ID"]]['windtex'] = float(damage['Windtex Estimation'])

    for f in ["size", "edge"]:
        gm_data = group_norm(m_data, f, 10)
        for gm in gm_data:
            m_data[gm][f] = gm_data[gm]
    if range_extend:
        for fea in all_feature:
            feature_range[fea] = [i for i in range(min(feature_range[fea]), 1 + max(feature_range[fea]) * 2)]

    return m_data