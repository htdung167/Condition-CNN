import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import get_file
import sys
from datetime import datetime
import os
import argparse
from utils.load_map import load_txt_to_dic

#-- Import wandb
import wandb
import tensorflow as tf
from wandb.keras import WandbCallback
wandb.init(project="ConditionCNN_15_11_2022", entity="thesis_uit_taidung")

def get_flow_from_dataframe(g, dataframe,image_shape=(224, 224),batch_size=128):
    while True:
        x_1 = g.next()
        yield [x_1[0], x_1[1][0], x_1[1][1]], x_1[1]
def train_BCNN(
    label, model, cbks, weights_path, direc, 
    target_size, batch, epochs, 
    train_datagen, train_df,
    val_datagen, val_df,
    x_column, y_column,
    TODAY):
    model.load_weights(weights_path, by_name=True)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=direc,
        # x_col="filepath",
        x_col=x_column,
        # y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        y_col=y_column,
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        # x_col="filepath",
        x_col=x_column,
        # y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        y_col=y_column,
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')
    try:
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = val_generator.n // val_generator.batch_size
        history = model.fit_generator(train_generator,
                            epochs=epochs,
                            validation_data=val_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_steps=STEP_SIZE_VALID,
                            callbacks=cbks)
        print("Finished training")
        #Save training as csv
        pd.DataFrame.from_dict(history.history).to_csv("./history/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'.csv',index=False)
    
        # plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig("./plots/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'_loss.png', bbox_inches='tight')
        # plt.show()

    except ValueError as v:
        print(v)

    # Saving the weights in the current directory
    model.save_weights("./weights/"+label+"_"+str(epochs)+"_epochs_"+TODAY+".h5")

def train_recurrent(
    label, model, cbks, weights_path, direc, 
    target_size, batch, epochs, 
    train_datagen, train_df,
    val_datagen, val_df,
    x_column, y_column,
    TODAY):
    #model.load_weights(weights_path, by_name=True)
    train = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=direc,
        # x_col="filepath",
        # y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        x_col=x_column,
        y_col=y_column,
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')
    val = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        # x_col="filepath",
        # y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        x_col=x_column,
        y_col=y_column,
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')

    train_generator = get_flow_from_dataframe(train,dataframe=train_df,image_shape=target_size,batch_size=batch)
    val_generator = get_flow_from_dataframe(val,dataframe=val_df,image_shape=target_size,batch_size=batch)
    try:
        STEP_SIZE_TRAIN = train.n // train.batch_size
        STEP_SIZE_VALID = val.n // val.batch_size
        history = model.fit_generator(train_generator,
                            epochs=epochs,
                            validation_data=val_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_steps=STEP_SIZE_VALID,
                            callbacks=cbks)
        print("Finished training")
        #Save training as csv
        pd.DataFrame.from_dict(history.history).to_csv("./history/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'.csv',index=False)

        # summarize history for loss
        plt.plot(history.history['master_output_loss'])
        plt.plot(history.history['val_master_output_loss'])
        plt.plot(history.history['sub_output_loss'])
        plt.plot(history.history['val_sub_output_loss'])
        plt.plot(history.history['article_output_loss'])
        plt.plot(history.history['val_article_output_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train master', 'val master', 'train sub', 'val sub', 'train article', 'val article'], loc='upper right')
        plt.savefig("./plots/"+label+"_"+str(epochs)+"_epochs_"+TODAY+"_loss.png", bbox_inches='tight')
        # plt.show()
    except ValueError as v:
        print(v)

    # Saving the weights in the current directory
    model.save_weights("./weights/"+label+"_"+str(epochs)+"_epochs_"+TODAY+".h5")     

def train_baseline(
    label, model, cbks, weights_path, direc, 
    target_size, batch, epochs, 
    train_datagen, train_df,
    val_datagen, val_df,
    x_column, y_column,
    TODAY):
    model.load_weights(weights_path, by_name=True)
    '''label is masterCategory, subCategory, or, articleType'''
    y = label
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=direc,
        # x_col="filepath",
        # y_col=y,
        x_col=x_column,
        y_col=y_column,
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        # x_col="filepath",
        # y_col=y,
        x_col=x_column,
        y_col=y_column,
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')
    try:
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = val_generator.n // val_generator.batch_size
        history = model.fit_generator(train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            epochs=epochs,
                            validation_data=val_generator,
                            validation_steps=STEP_SIZE_VALID,
                            callbacks=cbks)
        print("Finished training")
        #Save training as csv
        pd.DataFrame.from_dict(history.history).to_csv("./history/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'.csv',index=False)
    
        # plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig("./plots/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'_loss.png', bbox_inches='tight')
        # plt.show()

    except ValueError as v:
        print(v)

    # Saving the weights in the current directory
    model.save_weights("./weights/"+label+"_"+str(epochs)+"_epochs_"+TODAY+".h5")

def train_FineTuningDARTS(
    label, model, cbks, weights_path, direc, 
    target_size, batch, epochs, 
    train_datagen, train_df,
    val_datagen, val_df,
    x_column, y_column,
    TODAY):
    #model.load_weights(weights_path, by_name=True)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=direc,
        # x_col="filepath",
        x_col=x_column,
        # y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        y_col=y_column,
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        # x_col="filepath",
        x_col=x_column,
        # y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        y_col=y_column,
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')
    try:
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = val_generator.n // val_generator.batch_size
        history = model.fit_generator(train_generator,
                            epochs=epochs,
                            validation_data=val_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_steps=STEP_SIZE_VALID,
                            callbacks=cbks)
        print("Finished training")
        #Save training as csv
        pd.DataFrame.from_dict(history.history).to_csv("./history/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'.csv',index=False)
    
        # plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig("./plots/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'_loss.png', bbox_inches='tight')
        # plt.show()

    except ValueError as v:
        print(v)
    # Saving the weights in the current directory
    model.save_weights("./weights/"+label+"_"+str(epochs)+"_epochs_"+TODAY+".h5")
def main(opt):
    model_type, epochs, batch, data_path, imgsz, csv_train, csv_val, csv_test, master_column, sub_column, article_column, filepath_column = \
        opt.model, opt.epochs, opt.batch, opt.data_path, opt.imgsz, opt.csv_train, opt.csv_val, opt.csv_test, opt.master_column, opt.sub_column, opt.article_column, opt.filepath_column

    #Create working folder
    weights_folder = "./weights"
    if not os.path.exists(weights_folder):
        os.mkdir(weights_folder)
    plots_folder = "./plots"
    if not os.path.exists(plots_folder):
        os.mkdir(plots_folder)
    history_folder = "./history"
    if not os.path.exists(history_folder):
        os.mkdir(history_folder)
    
    #Loading data
    train_df = pd.read_csv(csv_train)
    val_df = pd.read_csv(csv_val)
    test_df = pd.read_csv(csv_test)

    # Add column one hot
    #master_column_one_hot = master_column + "OneHot"
    #sub_column_one_hot = sub_column + "OneHot"
    article_column_one_hot = article_column + "OneHot"

    if(model_type=='Recurrent' or model_type=='BCNN' or model_type=='Condition' or model_type=='ConditionPlus' or model_type=='ConditionB' or model_type=='FineTuningDARTS' ):

        # TODO: Map txt
        lblmapsub = {'Bags': 0, 'Belts': 1, 'Bottomwear': 2, 'Dress': 3, 'Eyewear': 4, 'Flip Flops': 5, 'Fragrance': 6, 'Headwear': 7, 'Innerwear': 8, 'Jewellery': 9, 'Lips': 10, 'Loungewear and Nightwear': 11, 'Nails': 12, 'Sandal': 13, 'Saree': 14, 'Shoes': 15, 'Socks': 16, 'Ties': 17, 'Topwear': 18, 'Wallets': 19, 'Watches': 20}
        lblmaparticle = {'Backpacks': 0, 'Belts': 1, 'Bra': 2, 'Briefs': 3, 'Capris': 4, 'Caps': 5, 'Casual Shoes': 6, 'Clutches': 7, 'Deodorant': 8, 'Dresses': 9, 'Earrings': 10, 'Flats': 11, 'Flip Flops': 12, 'Formal Shoes': 13, 'Handbags': 14, 'Heels': 15, 'Innerwear Vests': 16, 'Jackets': 17, 'Jeans': 18, 'Kurtas': 19, 'Kurtis': 20, 'Leggings': 21, 'Lipstick': 22, 'Nail Polish': 23, 'Necklace and Chains': 24, 'Nightdress': 25, 'Pendant': 26, 'Perfume and Body Mist': 27, 'Sandals': 28, 'Sarees': 29, 'Shirts': 30, 'Shorts': 31, 'Socks': 32, 'Sports Shoes': 33, 'Sunglasses': 34, 'Sweaters': 35, 'Sweatshirts': 36, 'Ties': 37, 'Tops': 38, 'Track Pants': 39, 'Trousers': 40, 'Tshirts': 41, 'Tunics': 42, 'Wallets': 43, 'Watches': 44}
        lblmapmaster = {'Accessories': 0, 'Apparel': 1, 'Footwear': 2, 'Personal Care': 3}
        #lblmapsub, _ = load_txt_to_dic("./data/map_level_1.txt")
        #lblmaparticle, _ = load_txt_to_dic("./data/map_level_2.txt")
        #lblmapmaster, _ = load_txt_to_dic("./data/map_level_0.txt")

        #Map classes
        #train_df[master_column].replace(lblmapmaster,inplace=True)
        #test_df[master_column].replace(lblmapmaster,inplace=True)
        #val_df[master_column].replace(lblmapmaster,inplace=True)

        #train_df[sub_column].replace(lblmapsub,inplace=True)
        #test_df[sub_column].replace(lblmapsub,inplace=True)
        #val_df[sub_column].replace(lblmapsub,inplace=True)

        train_df[article_column].replace(lblmaparticle,inplace=True)
        test_df[article_column].replace(lblmaparticle,inplace=True)
        val_df[article_column].replace(lblmaparticle,inplace=True)

        #Convert the 3 labels to one hots in train, test, val
        # onehot_master = to_categorical(train_df[master_column].values)
        # train_df[master_column_one_hot] = onehot_master.tolist()
        # onehot_master = to_categorical(val_df[master_column].values)
        # val_df[master_column_one_hot] = onehot_master.tolist()
        # onehot_master = to_categorical(test_df[master_column].values)
        # test_df[master_column_one_hot] = onehot_master.tolist()

        # onehot_master = to_categorical(train_df[sub_column].values)
        # train_df[sub_column_one_hot] = onehot_master.tolist()
        # onehot_master = to_categorical(val_df[sub_column].values)
        # val_df[sub_column_one_hot] = onehot_master.tolist()
        # onehot_master = to_categorical(test_df[sub_column].values)
        # test_df[sub_column_one_hot] = onehot_master.tolist()

        onehot_master = to_categorical(train_df[article_column].values)
        train_df[article_column_one_hot] = onehot_master.tolist()
        onehot_master = to_categorical(val_df[article_column].values)
        val_df[article_column_one_hot] = onehot_master.tolist()
        onehot_master = to_categorical(test_df[article_column].values)
        test_df[article_column_one_hot] = onehot_master.tolist()
    else:
        raise NotImplementedError(f"Model type {model_type} is not supported.")

    #----------get VGG16 pre-trained weights--------
    #WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    # #weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
    #                         WEIGHTS_PATH,
    #                         cache_subdir='./weights')
    weights_path =""
    #-- Wandb Config
    wandb.config = {
        "learning_rate": 0.001,
        "epochs": epochs,
        "batch_size": batch
    }

    #----------globals---------
    direc = data_path
    target_size=(imgsz, imgsz)
    input_shape = target_size + (3,)
    TODAY = str(datetime.date(datetime.now()))

    master_classes = len(lblmapmaster)
    sub_classes = len(lblmapsub)
    art_classes = len(lblmaparticle)

    #Do additional transformations to support BatchNorm, Featurewise center and scal so each feature roughly N(0,1)
    #Try with and without rescale
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    samplewise_center=True,
                                    samplewise_std_normalization=True)
    val_datagen = ImageDataGenerator(rescale=1. / 255,
                                    samplewise_center=True,
                                    samplewise_std_normalization=True)

    if(model_type == 'Recurrent'):
        from model.RecurrentBranching import RecurrentTrain
        recurrent = RecurrentTrain(
            model_type, 
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            )
        model = recurrent.model
        cbks = recurrent.cbks
        x_column = filepath_column
        y_column = [master_column_one_hot, sub_column_one_hot, article_column_one_hot]
        train_recurrent(
            model_type, model, cbks, weights_path, direc, 
            target_size, batch, epochs, 
            train_datagen, train_df,
            val_datagen, val_df,
            x_column, y_column,
            TODAY
        )
    elif(model_type=='Condition'):
        from model.ConditionCNN import ConditionTrain
        condition = ConditionTrain(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            )
        model = condition.model
        cbks = condition.cbks

        #-- Add Wandb checkpoint
        #cbks = [*cbks, WandbCallback()]
        
        x_column = filepath_column
        y_column = [master_column_one_hot, sub_column_one_hot, article_column_one_hot]
        train_recurrent(
            model_type, model, cbks, weights_path, direc, 
            target_size, batch, epochs, 
            train_datagen, train_df,
            val_datagen, val_df,
            x_column, y_column,
            TODAY
        )
    elif(model_type=='ConditionB'):
        from model.ConditionCNNB import ConditionTrain
        condition = ConditionTrain(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            )
        model = condition.model
        cbks = condition.cbks
        x_column = filepath_column
        y_column = [master_column_one_hot, sub_column_one_hot, article_column_one_hot]
        train_recurrent(
            model_type, model, cbks, weights_path, direc, 
            target_size, batch, epochs, 
            train_datagen, train_df,
            val_datagen, val_df,
            x_column, y_column,
            TODAY
        )
    elif(model_type=='ConditionPlus'):
        from model.ConditionCNNPlus import ConditionPlusTrain
        condition = ConditionPlusTrain(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            )
        model = condition.model
        cbks = condition.cbks
        x_column = filepath_column
        y_column = [master_column_one_hot, sub_column_one_hot, article_column_one_hot]
        train_recurrent(
            model_type, model, cbks, weights_path, direc, 
            target_size, batch, epochs, 
            train_datagen, train_df,
            val_datagen, val_df,
            x_column, y_column,
            TODAY
        )
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
        cbks = bcnn.cbks
        x_column = filepath_column
        y_column = [master_column_one_hot, sub_column_one_hot, article_column_one_hot]
        train_BCNN(
            model_type, model, cbks, weights_path, direc, 
            target_size, batch, epochs, 
            train_datagen, train_df,
            val_datagen, val_df,
            x_column, y_column,
            TODAY
        )
    elif(model_type == 'articleType'):
        from model.articleType import ArticleType
        articletype = ArticleType(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            )
        model = articletype.model
        cbks = articletype.cbks
        x_column=filepath_column
        y_column=article_column_one_hot
        train_baseline(
            model_type, model, cbks, weights_path, direc, 
            target_size, batch, epochs, 
            train_datagen, train_df,
            val_datagen, val_df,
            x_column, y_column,
            TODAY
        )
    elif(model_type == 'subCategory'):
        from model.subCategory import SubCategory
        subcategory = SubCategory(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            )
        model = subcategory.model
        cbks = subcategory.cbks
        x_column=filepath_column
        y_column=sub_column_one_hot
        train_baseline(
            model_type, model, cbks, weights_path, direc, 
            target_size, batch, epochs, 
            train_datagen, train_df,
            val_datagen, val_df,
            x_column, y_column,
            TODAY
        )
    elif(model_type== 'FineTuningDARTS'):
        from model.Fine_tunning_DARTS import FineTuningDARTSTrain
        FineTuningDARTS= FineTuningDARTSTrain(
            model_type,
            art_classes= art_classes,
            input_image_shape= input_shape
        )
        model= FineTuningDARTS.model
        cbks = FineTuningDARTS.cbks
                #-- Add Wandb checkpoint
        cbks = [*cbks, WandbCallback()]

        x_column = filepath_column
        y_column = [article_column_one_hot]
        train_FineTuningDARTS(
            model_type, model, cbks, weights_path, direc, 
            target_size, batch, epochs, 
            train_datagen, train_df,
            val_datagen, val_df,
            x_column, y_column,
            TODAY
        )


    else:
        #masterCategory
        from model.masterCategory import MasterCategory
        mastercategory = MasterCategory(
            model_type,
            master_classes=master_classes, 
            sub_classes=sub_classes, 
            art_classes=art_classes, 
            input_img_shape=input_shape
            )
        model = mastercategory.model
        cbks = mastercategory.cbks
        x_column=filepath_column
        y_column=master_column_one_hot
        train_baseline(
            model_type, model, cbks, weights_path, direc, 
            target_size, batch, epochs, 
            train_datagen, train_df,
            val_datagen, val_df,
            x_column, y_column,
            TODAY
        )
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Recurrent', help='Model type')
    parser.add_argument('--epochs', type=int, default=20, help='Epochs for training')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=224, help='Image size')
    parser.add_argument('--data_path', type=str, default='./data/fashion-dataset/images/', help='Image dataset path')
    parser.add_argument('--csv_train', type=str, help='Train info path')
    parser.add_argument('--csv_val', type=str, help='Validation info path')
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
