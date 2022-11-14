import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import get_file
import sys
from datetime import datetime
from os import path
import tensorflow.keras.backend as K
import argparse
import os
from utils.load_map import load_txt_to_dic


def test_multi(
    label, model, 
    target_size, batch,
    direc,
    weights_path, 
    x_column, y_column,
    test_datagen, test_df
):
    model.load_weights(weights_path, by_name=True)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=direc,
        # x_col="filepath",
        # y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        x_col=x_column, y_col=y_column,
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')

    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    #x can be a generator returning retunring (inputs, targets)
    #if x is a generator y should not be specified
    score = model.evaluate(x=test_generator,
        steps=STEP_SIZE_TEST)

    print(f"score is {score}")
    return score

def test_articleType(
    label, model, 
    target_size, batch,
    direc,
    weights_path, 
    x_column, y_column,
    test_datagen, test_df
):
    model.load_weights(weights_path, by_name=True)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=direc,
        # x_col="filepath",
        # y_col='articleType',
        x_col=x_column, y_col=y_column,
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    score = model.evaluate(x=test_generator,
        steps=STEP_SIZE_TEST)

    print(f"score is {score}")
    return score

def test_subCategory(
    label, model, 
    target_size, batch,
    direc,
    weights_path, 
    x_column, y_column,
    test_datagen, test_df
):
    model.load_weights(weights_path, by_name=True)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=direc,
        # x_col="filepath",
        # y_col='subCategory',
        x_col=x_column, y_col=y_column,
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    score = model.evaluate(x=test_generator,
        steps=STEP_SIZE_TEST)

    print(f"score is {score}")
    return score

def test_masterCategory(
    label, model, 
    target_size, batch,
    direc,
    weights_path, 
    x_column, y_column,
    test_datagen, test_df
):
    model.load_weights(weights_path, by_name=True)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=direc,
        # x_col="filepath",
        # y_col='masterCategory',
        x_col=x_column, y_col=y_column,
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    score = model.evaluate(x=test_generator,
        steps=STEP_SIZE_TEST)

    print(f"score is {score}")
    return score

def main(opt):
    model_type, weights_path, batch, data_path, imgsz, csv_test, master_column, sub_column, article_column, filepath_column = \
        opt.model, opt.weights_path, opt.batch, opt.data_path, opt.imgsz, opt.csv_test, opt.master_column, opt.sub_column, opt.article_column, opt.filepath_column

    #Create working folder
    testing_folder = "./testing"
    if not os.path.exists(testing_folder):
        os.mkdir(testing_folder)
    
    #Loading test data
    test_df = pd.read_csv(csv_test)

    #check if results csv exists. if it does not then create it. append each result test run as a new label 
    #row contains the label (BCNN, Recurrent etc), the weights path, the three test accuracies (or 1 if it is a baseline cnn model)
    # and the timestamp when it was tested
    exists = os.path.exists("./testing/test_results.csv")
    if(not exists):
        dtypes = np.dtype([
            ('Model', str),
            ('Weights Path', str),
            ('masterCategory Accuracy %', np.float64),
            ('subCategory Accuracy %', np.float64),
            ('articleType Accuracy %', np.float64),
            ('Trainable params', np.float64),
            ('Timestamp', np.datetime64),
            ])
        #df = pd.DataFrame(columns=['Model','Weights Path', 'masterCategory Accuracy %','subCategory Accuracy %','articleType Accuracy %','Timestamp'])
        data = np.empty(0, dtype=dtypes)
        df = pd.DataFrame(data)
    else:
        types = {'Model':str,'Weights Path': str, 'masterCategory Accuracy %': np.float64, 'subCategory Accuracy %': np.float64,
        'articleType Accuracy %': np.float64, 'Trainable params': np.float64}
        df = pd.read_csv("./testing/test_results.csv",dtype=types, parse_dates=['Timestamp'])


    # Add column one hot
    master_column_one_hot = master_column + "OneHot"
    sub_column_one_hot = sub_column + "OneHot"
    article_column_one_hot = article_column + "OneHot"

    if(model_type=='Recurrent' or model_type=='BCNN' or model_type=='Condition'or model_type=='ConditionPlus'or model_type=='ConditionB'):

        # lblmapsub = {'Bags': 0, 'Belts': 1, 'Bottomwear': 2, 'Dress': 3, 'Eyewear': 4, 'Flip Flops': 5, 'Fragrance': 6, 'Headwear': 7, 'Innerwear': 8, 'Jewellery': 9, 'Lips': 10, 'Loungewear and Nightwear': 11, 'Nails': 12, 'Sandal': 13, 'Saree': 14, 'Shoes': 15, 'Socks': 16, 'Ties': 17, 'Topwear': 18, 'Wallets': 19, 'Watches': 20}
        # lblmaparticle = {'Backpacks': 0, 'Belts': 1, 'Bra': 2, 'Briefs': 3, 'Capris': 4, 'Caps': 5, 'Casual Shoes': 6, 'Clutches': 7, 'Deodorant': 8, 'Dresses': 9, 'Earrings': 10, 'Flats': 11, 'Flip Flops': 12, 'Formal Shoes': 13, 'Handbags': 14, 'Heels': 15, 'Innerwear Vests': 16, 'Jackets': 17, 'Jeans': 18, 'Kurtas': 19, 'Kurtis': 20, 'Leggings': 21, 'Lipstick': 22, 'Nail Polish': 23, 'Necklace and Chains': 24, 'Nightdress': 25, 'Pendant': 26, 'Perfume and Body Mist': 27, 'Sandals': 28, 'Sarees': 29, 'Shirts': 30, 'Shorts': 31, 'Socks': 32, 'Sports Shoes': 33, 'Sunglasses': 34, 'Sweaters': 35, 'Sweatshirts': 36, 'Ties': 37, 'Tops': 38, 'Track Pants': 39, 'Trousers': 40, 'Tshirts': 41, 'Tunics': 42, 'Wallets': 43, 'Watches': 44}
        # lblmapmaster = {'Accessories': 0, 'Apparel': 1, 'Footwear': 2, 'Personal Care': 3}

        lblmapsub, _ = load_txt_to_dic("./data/map_level_1.txt")
        lblmaparticle, _ = load_txt_to_dic("./data/map_level_2.txt")
        lblmapmaster, _ = load_txt_to_dic("./data/map_level_0.txt")

        #Map classes
        test_df[master_column].replace(lblmapmaster,inplace=True)
        test_df[sub_column].replace(lblmapsub,inplace=True)
        test_df[article_column].replace(lblmaparticle,inplace=True)

        #Convert the 3 labels to one hots in train, test, val
        onehot_master = to_categorical(test_df[master_column].values)
        test_df[master_column_one_hot] = onehot_master.tolist()

        onehot_master = to_categorical(test_df[sub_column].values)
        test_df[sub_column_one_hot] = onehot_master.tolist()

        onehot_master = to_categorical(test_df[article_column].values)
        test_df[article_column_one_hot] = onehot_master.tolist()
    else:
        raise NotImplementedError(f"Model type {model_type} is not supported.")

    #----------globals---------
    direc = data_path
    target_size=(imgsz, imgsz)
    input_shape = target_size + (3,)
    TODAY = str(datetime.date(datetime.now()))

    master_classes = len(lblmapmaster)
    sub_classes = len(lblmapsub)
    art_classes = len(lblmaparticle)

    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True)
    
    score = 0
    params = 0
    masterCategory_accuracy = np.nan
    subCategory_accuracy = np.nan
    articleType_accuracy = np.nan

    if(model_type == 'Recurrent'):
        from model.RecurrentBranching import RecurrentTest
        model = RecurrentTest(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            ).model
        # score = test_multi(model_type, model)
        x_column=filepath_column
        y_column=[master_column_one_hot, sub_column_one_hot, article_column_one_hot]
        score = test_multi(
            model_type, model, 
            target_size, batch,
            direc,
            weights_path, 
            x_column, y_column,
            test_datagen, test_df
        )
        params= np.sum([K.count_params(w) for w in model.trainable_weights])
        masterCategory_accuracy = score[4]
        subCategory_accuracy = score[5]
        articleType_accuracy = score[6]
    elif(model_type == 'Condition'):
        from model.ConditionCNN import ConditionTest
        model = ConditionTest(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
        ).model
        x_column=filepath_column
        y_column=[master_column_one_hot, sub_column_one_hot, article_column_one_hot]
        # score = test_multi(model_type, model)
        score = test_multi(
            model_type, model, 
            target_size, batch,
            direc,
            weights_path, 
            x_column, y_column,
            test_datagen, test_df
        )

        params= np.sum([K.count_params(w) for w in model.trainable_weights])
        masterCategory_accuracy = score[4]
        subCategory_accuracy = score[5]
        articleType_accuracy = score[6]
    elif(model_type == 'ConditionB'):
        from model.ConditionCNNB import ConditionTest
        model = ConditionTest(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            ).model
        x_column=filepath_column
        y_column=[master_column_one_hot, sub_column_one_hot, article_column_one_hot]
        # score = test_multi(model_type, model)
        score = test_multi(
            model_type, model, 
            target_size, batch,
            direc,
            weights_path, 
            x_column, y_column,
            test_datagen, test_df
        )

        params= np.sum([K.count_params(w) for w in model.trainable_weights])
        masterCategory_accuracy = score[4]
        subCategory_accuracy = score[5]
        articleType_accuracy = score[6]
    elif(model_type == 'ConditionPlus'):
        from model.ConditionCNNPlus import ConditionPlusTest
        model = ConditionPlusTest(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            ).model
        x_column=filepath_column
        y_column=[master_column_one_hot, sub_column_one_hot, article_column_one_hot]
        # score = test_multi(model_type, model)
        score = test_multi(
            model_type, model, 
            target_size, batch,
            direc,
            weights_path, 
            x_column, y_column,
            test_datagen, test_df
        )
        params= np.sum([K.count_params(w) for w in model.trainable_weights])
        masterCategory_accuracy = score[4]
        subCategory_accuracy = score[5]
        articleType_accuracy = score[6]
    elif(model_type=='BCNN'):
        from model.BCNN import BCNN
        bcnn = BCNN(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            )
        model = bcnn.model
        # score = test_multi(model_type, model)
        x_column=filepath_column
        y_column=[master_column_one_hot, sub_column_one_hot, article_column_one_hot]
        # score = test_multi(model_type, model)
        score = test_multi(
            model_type, model, 
            target_size, batch,
            direc,
            weights_path, 
            x_column, y_column,
            test_datagen, test_df
        )
        params= np.sum([K.count_params(w) for w in model.trainable_weights])

        #score [0:4] are the losses for the branches
        masterCategory_accuracy = score[4]
        subCategory_accuracy = score[5]
        articleType_accuracy = score[6]

    elif(model_type == 'articleType'):
        from model.articleType import ArticleType
        model = ArticleType(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            ).model
        x_column=filepath_column
        y_column=article_column
        # score = test_articleType(model_type, model)
        score = test_articleType(
            model_type, model, 
            target_size, batch,
            direc,
            weights_path, 
            x_column, y_column,
            test_datagen, test_df
        )

        params= np.sum([K.count_params(w) for w in model.trainable_weights])
        articleType_accuracy = score[1]

    elif(model_type == 'subCategory'):
        from model.subCategory import SubCategory
        model = SubCategory(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            ).model
        x_column=filepath_column
        y_column=sub_column
        # score = test_subCategory(model_type,model)
        score = test_subCategory(
            model_type, model, 
            target_size, batch,
            direc,
            weights_path, 
            x_column, y_column,
            test_datagen, test_df
        )
        params= np.sum([K.count_params(w) for w in model.trainable_weights])
        subCategory_accuracy = score[1]
    else:
        #masterCategory
        from model.masterCategory import MasterCategory
        model = MasterCategory(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            ).model
        x_column=filepath_column
        y_column=master_column
        # score = test_masterCategory(model_type, model)
        score = test_masterCategory(
            model_type, model, 
            target_size, batch,
            direc,
            weights_path, 
            x_column, y_column,
            test_datagen, test_df
        )
        params= np.sum([K.count_params(w) for w in model.trainable_weights])
        masterCategory_accuracy = score[1]

        
    df.loc[df.index.max()+1] = [model_type, weights_path, masterCategory_accuracy, subCategory_accuracy, articleType_accuracy, params,np.datetime64('now')]
    df.to_csv("./testing/test_results.csv", index=False)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Recurrent', help='Model type')
    parser.add_argument('--weights_path', type=str, help='Weights path')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=224, help='Image size')
    parser.add_argument('--data_path', type=str, default='./data/fashion-dataset/images/', help='Image dataset path')
    parser.add_argument('--csv_test', type=str, help='Test info path')
    parser.add_argument('--master_column', type=str, default='masterCategory', help="Name of master level column in info csv file")
    parser.add_argument('--sub_column', type=str, default='subCategory', help="Name of sub level column in info csv file")
    parser.add_argument('--article_column', type=str, default='articleType', help="Name of article level column in info csv file")
    parser.add_argument('--filepath_column', type=str, default='filepath', help="Name of filepath column in info csv file")
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)