from sklearn.model_selection import train_test_split
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.load_map import load_txt_to_dic

def ReadFileCsv(fileCsv):
    df = pd.read_csv(fileCsv)
    return df

def FormTree(lst):
    tree = {}
    for item in lst:
        currTree = tree
        for key in item:
            if key not in currTree:
                currTree[key] = {}
            currTree = currTree[key]
    return tree

def GetNameAndNumberOfCatesInEachLevel(fileCsv, show=False, lstColumnNameLevel=[]):
    df = ReadFileCsv(fileCsv)
    dic = {}
    lstBranch = []
    lstNumberCateEachLevel = []
    lstNameEachLevel = []
    # Name
    # ChildrenCount
    # ChildCategories
    for i, nameLevel in enumerate(lstColumnNameLevel):
        subLst = df.get(nameLevel).tolist()
        lstBranch.append(subLst)
        lstNumberCateEachLevel.append(len(list(set(subLst))))
        lstNameEachLevel.append(list(set(subLst)))
    lstBranch = list(set(list(zip(*lstBranch)))) # get unique tuple branch
    tree = FormTree(lstBranch)
    if show:
        print(tree)
        print(lstNumberCateEachLevel)
        print(lstNameEachLevel)
        print(lstBranch)


    return tree, lstNumberCateEachLevel, lstNameEachLevel, lstBranch


def GetTotalNumberOfImages(fileCsv, show=False):
    df = ReadFileCsv(fileCsv)
    result = len(df)
    if show:
        print(f"Tổng ảnh: {result}")
    return result
    

def GetNumberOfImagesEachCateEachLevel(fileCsv, show=False, lstColumnNameLevel=[]):
    df = ReadFileCsv(fileCsv)
    _, _, lstNameEachLevel, _ = GetNameAndNumberOfCatesInEachLevel(fileCsv, lstColumnNameLevel=lstColumnNameLevel)
    # print(lstNameEachLevel)
    result = []
    for i, cates in enumerate(lstNameEachLevel):
        dic = {}
        for cate in cates:
            # print(lstColumnNameLevel[i], cate)
            dff = df[df[lstColumnNameLevel[i]] == cate]
            dic[cate] = len(dff)
        result.append(dic)
    if show:
        print(result)
    return result


def GetMaximumNumberOfImagesEachLevel(fileCsv, show=False, lstColumnNameLevel=[]):
    result = {}
    numberOfImagesEachCateEachLevel = GetNumberOfImagesEachCateEachLevel(fileCsv, lstColumnNameLevel=lstColumnNameLevel)
    for dic_level in numberOfImagesEachCateEachLevel:
        key_max = max(dic_level, key=dic_level.get)
        result[key_max]=dic_level[key_max]
    if show:
        print(result)
    return result

def GetMinimumNumberOfImagesEachLevel(fileCsv, show=False, lstColumnNameLevel=[]):
    result = {}
    numberOfImagesEachCateEachLevel = GetNumberOfImagesEachCateEachLevel(fileCsv, lstColumnNameLevel=lstColumnNameLevel)
    for dic_level in numberOfImagesEachCateEachLevel:
        key_max = min(dic_level, key=dic_level.get)
        result[key_max]=dic_level[key_max]
    if show:
        print(result)
    return result

def GetAverageNumberOfImagesEachLevel(fileCsv, show=False, lstColumnNameLevel=[]):
    result = []
    numberOfImagesEachCateEachLevel = GetNumberOfImagesEachCateEachLevel(fileCsv, lstColumnNameLevel=lstColumnNameLevel)
    for i, dic_level in enumerate(numberOfImagesEachCateEachLevel):
        lenn = len(dic_level)
        summ = sum(list(dic_level.values()))
        print(summ, lenn)
        result.append(summ // lenn)
    if show:
        print(result)
    return result

def Visualizer():
    pass

def Save():
    pass

def MapIdToLabelShopee(
    fileCsv,
    lstColumnNameLevelId=["MasterCategoryId", "SubCategoryId", "ArticalTypeId"], 
    lstColumnNameLevelLabel=["MasterCategory", "SubCategory", "ArticalType"]):
    """Map cateId to Label"""
    df = ReadFileCsv(fileCsv)
    tree, lstNumberCateEachLevel, lstNameEachLevel, lstBranch = \
        GetNameAndNumberOfCatesInEachLevel(fileCsv, show=False, lstColumnNameLevel=lstColumnNameLevelId)
    # print(lstNameEachLevel)
    result = []
    for i, lstLevel in enumerate(lstNameEachLevel):
        dic = dict()
        for id in lstLevel:
            label = df[df[lstColumnNameLevelId[i]] == id][lstColumnNameLevelLabel[i]]
            label = list(set(label.tolist()))
            dic[str(id)] = label[0]
            # print(type(id))
        result.append(dic)
    # print(result)
    return result

def SaveMapToTxt(fileCsv, save_path = "./data", name = "map_level_"):
    """Save Map to Txt. Ex: 1919191 0 Áo Khoác"""
    mapIdToLabel =  MapIdToLabelShopee(fileCsv)
    for i, labelMap in enumerate(mapIdToLabel):
        save_file = os.path.join(save_path, name + str(i) + ".txt")
        with open(save_file, 'w', encoding='utf-8') as f:
            for j, (k, v) in enumerate(labelMap.items()):
                txt = str(k) + " " + str(j) + " " + str(v)
                f.write(txt + "\n")




def SplitData(
    fileCsv, 
    val_ratio=0.1, 
    test_ratio=0.25, 
    save_path = "./data", 
    train_name="shopee_fashion_train", 
    val_name="shopee_fashion_val", 
    test_name="shopee_fashion_test", 
    y_col="ArticalTypeId"):

    df = pd.read_csv(fileCsv, index_col=0)

    # print(df[lstColumnConvert].info())
    df_train = []
    df_val = []
    df_test = []
    val_ratio_after_split_test = val_ratio / (1 - test_ratio)

    groups = df.groupby(y_col)

    for label in df[y_col].unique():
        group = groups.get_group(label)
        train, test = train_test_split(group, test_size=test_ratio, random_state=0)
        train, val = train_test_split(train, test_size=val_ratio_after_split_test, random_state=0)

        if len(df_train)==0:
            df_train = train
        else:
            df_train = pd.concat([df_train, train], ignore_index=True)
        if len(df_val)==0:
            df_val = val
        else:
            df_val = pd.concat([df_val, val], ignore_index=True)
        if len(df_test)==0:
            df_test = test
        else:
            df_test = pd.concat([df_test, test], ignore_index=True)
    
    df_train.to_csv(os.path.join(save_path, train_name + ".csv"))
    df_val.to_csv(os.path.join(save_path, val_name + ".csv"))
    df_test.to_csv(os.path.join(save_path, test_name + ".csv"))

def ConvertPath(fileCsv, save_file = "data/shopee_fashion_dataset2.csv"):
    df = pd.read_csv(fileCsv, index_col=0)
    print(df.head())
    df["FilePath"] = df["FilePath"].apply(lambda x: x.replace("\\", "/"))
    print(df.head())
    df.to_csv(save_file)
    
def LoadDic():
    lblmapsub, _ = load_txt_to_dic("./data/map_level_1.txt")
    lblmaparticle, _ = load_txt_to_dic("./data/map_level_2.txt")
    lblmapmaster, _ = load_txt_to_dic("./data/map_level_0.txt")
    print(lblmapmaster)
    print(lblmapsub)
    print(lblmaparticle)


if __name__=="__main__":
    # Split dataset
    fileCsv = "data/shopee_fashion_dataset2.csv"
    # ConvertPath(fileCsv)

    # SplitData(
    #     fileCsv, 
    #     val_ratio=0.1, 
    #     test_ratio=0.25, 
    #     save_path = "./data", 
    #     train_name="shopee_fashion_train", 
    #     val_name="shopee_fashion_val", 
    #     test_name="shopee_fashion_test", 
    #     y_col="ArticalTypeId")
    # Test(fileCsv)
    # MapIdToLabelShopee(fileCsv)
    # SaveMapToTxt(fileCsv)

    # print("All")
    # GetNumberOfImagesEachCateEachLevel(fileCsv, show=True, lstColumnNameLevel=["MasterCategoryId", "SubCategoryId", "ArticalTypeId"])
    # print("Train")
    # GetNumberOfImagesEachCateEachLevel(fileCsv="data/shopee_fashion_train.csv", show=True, lstColumnNameLevel=["MasterCategoryId", "SubCategoryId", "ArticalTypeId"])
    # print("Val")
    # GetNumberOfImagesEachCateEachLevel(fileCsv="data/shopee_fashion_val.csv", show=True, lstColumnNameLevel=["MasterCategoryId", "SubCategoryId", "ArticalTypeId"])
    # print("Test")
    # GetNumberOfImagesEachCateEachLevel(fileCsv="data/shopee_fashion_test.csv", show=True, lstColumnNameLevel=["MasterCategoryId", "SubCategoryId", "ArticalTypeId"])

    LoadDic()